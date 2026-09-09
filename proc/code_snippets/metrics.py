"""
dfx.metrics — evaluation that is comparable to the deepfake literature.

Fixes three problems in the previous setup:
  1. Reports video-level AUC (balance-invariant) as the PRIMARY metric.
  2. Reports F1 for BOTH class conventions, so "F1=0.78" is never ambiguous.
  3. Adds bootstrap CIs + paired Wilcoxon, covering the reviewer's
     "no significance testing" objection without inventing new runs.

Convention used throughout: label 1 == FAKE, label 0 == REAL.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score,
                             confusion_matrix)
from scipy.stats import wilcoxon


# ----------------------------------------------------------------------
# video-level aggregation
# ----------------------------------------------------------------------
def aggregate_clips(clip_scores, method="mean", k=5):
    """Collapse many clip scores for ONE video into a single video score.

    ForgeryNet's own tables show multi-crop beats single-crop by ~+3.2 AUC
    with no architecture change, so this function is where a large part of
    your free performance lives.

    method: mean | max | topk_mean | logit_mean
    """
    s = np.asarray(clip_scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("empty clip_scores")
    if method == "mean":
        return float(s.mean())
    if method == "max":
        return float(s.max())
    if method == "topk_mean":
        kk = int(min(max(k, 1), s.size))
        return float(np.sort(s)[-kk:].mean())
    if method == "logit_mean":
        eps = 1e-6
        p = np.clip(s, eps, 1 - eps)
        return float(1.0 / (1.0 + np.exp(-np.log(p / (1 - p)).mean())))
    raise ValueError(f"unknown aggregation: {method}")


def video_scores_from_clips(clip_table, method="mean", k=5):
    """clip_table: {video_id: [clip_score, ...]} -> ({vid: score}, order)."""
    out = {v: aggregate_clips(s, method, k) for v, s in clip_table.items()}
    return out


# ----------------------------------------------------------------------
# metric bundle
# ----------------------------------------------------------------------
@dataclass
class Report:
    n: int
    n_real: int
    n_fake: int
    pct_real: float
    threshold: float
    auc: float
    ap_fake: float
    accuracy: float
    balanced_accuracy: float
    f1_fake: float          # positive class = FAKE
    f1_real: float          # positive class = REAL  (what the old paper reported)
    f1_macro: float
    precision_fake: float
    recall_fake: float
    precision_real: float
    recall_real: float
    tn_real_real: int
    fp_real_fake: int
    fn_fake_real: int
    tp_fake_fake: int

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        return (f"n={self.n} ({self.pct_real:.1f}% real) thr={self.threshold:.3f}\n"
                f"  AUC              {self.auc:.4f}   <-- primary, balance-invariant\n"
                f"  AP(fake)         {self.ap_fake:.4f}\n"
                f"  accuracy         {self.accuracy:.4f}\n"
                f"  balanced acc     {self.balanced_accuracy:.4f}\n"
                f"  F1 (fake=pos)    {self.f1_fake:.4f}\n"
                f"  F1 (real=pos)    {self.f1_real:.4f}   <-- old paper's number\n"
                f"  F1 macro         {self.f1_macro:.4f}\n"
                f"  confusion  RR={self.tn_real_real} RF={self.fp_real_fake} "
                f"FR={self.fn_fake_real} FF={self.tp_fake_fake}")


def evaluate(y_true, y_score, threshold=0.5):
    """y_true: 1=fake 0=real. y_score: P(fake)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score length mismatch")
    y_pred = (y_score >= threshold).astype(int)

    n_fake = int(y_true.sum())
    n_real = int((1 - y_true).sum())
    single_class = (n_fake == 0 or n_real == 0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return Report(
        n=len(y_true), n_real=n_real, n_fake=n_fake,
        pct_real=100.0 * n_real / max(len(y_true), 1),
        threshold=float(threshold),
        auc=float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        ap_fake=float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1_fake=float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f1_real=float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_fake=float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        recall_fake=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        precision_real=float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        recall_real=float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        tn_real_real=int(tn), fp_real_fake=int(fp),
        fn_fake_real=int(fn), tp_fake_fake=int(tp),
    )


def best_threshold(y_true, y_score, criterion="youden"):
    """Pick an operating point on VALIDATION only, then freeze it for test."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    cand = np.unique(y_score)
    if cand.size > 2000:
        cand = np.quantile(y_score, np.linspace(0, 1, 2000))
    best, best_v = 0.5, -np.inf
    for t in cand:
        pred = (y_score >= t).astype(int)
        if criterion == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
            v = tpr - fpr
        elif criterion == "f1_macro":
            v = f1_score(y_true, pred, average="macro", zero_division=0)
        elif criterion == "balanced_acc":
            v = balanced_accuracy_score(y_true, pred)
        else:
            raise ValueError(criterion)
        if v > best_v:
            best_v, best = v, float(t)
    return best


# ----------------------------------------------------------------------
# uncertainty: bootstrap CI + paired significance
# ----------------------------------------------------------------------
def bootstrap_ci(y_true, y_score, metric="auc", n_boot=2000, alpha=0.05, seed=42):
    """Stratified bootstrap CI. Gives a confidence interval from a SINGLE run,
    which is not a substitute for multi-seed std but is honest and cheap."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    idx_r = np.where(y_true == 0)[0]
    idx_f = np.where(y_true == 1)[0]
    if len(idx_r) == 0 or len(idx_f) == 0:
        return (float("nan"),) * 3

    def _m(t, s):
        if metric == "auc":            return roc_auc_score(t, s)
        if metric == "ap":             return average_precision_score(t, s)
        if metric == "balanced_acc":   return balanced_accuracy_score(t, (s >= 0.5).astype(int))
        if metric == "f1_macro":       return f1_score(t, (s >= 0.5).astype(int), average="macro", zero_division=0)
        raise ValueError(metric)

    point = _m(y_true, y_score)
    vals = []
    for _ in range(n_boot):
        br = rng.choice(idx_r, len(idx_r), replace=True)
        bf = rng.choice(idx_f, len(idx_f), replace=True)
        b = np.concatenate([br, bf])
        try:
            vals.append(_m(y_true[b], y_score[b]))
        except ValueError:
            pass
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def seed_summary(values):
    """mean +/- std across seeds, formatted for a paper table."""
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "n": int(v.size), "fmt": f"{v.mean():.4f} ± {v.std(ddof=1) if v.size>1 else 0.0:.4f}"}


def paired_test(a, b, alt="two-sided"):
    """Paired Wilcoxon signed-rank across matched runs (same seeds/folds).

    This is what the reviewer asked for. Feed it per-seed or per-subset
    metric values for model A vs model B.
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired_test needs equal-length matched samples")
    d = a - b
    out = {"n_pairs": int(a.size), "mean_diff": float(d.mean()),
           "median_diff": float(np.median(d))}
    if np.allclose(d, 0):
        out.update(stat=float("nan"), p=1.0, note="identical samples")
        return out
    if a.size < 6:
        out["note"] = "n<6: Wilcoxon underpowered, report descriptively"
    try:
        st, p = wilcoxon(a, b, alternative=alt, zero_method="wilcox")
        out.update(stat=float(st), p=float(p))
    except ValueError as e:
        out.update(stat=float("nan"), p=float("nan"), note=str(e))
    return out
