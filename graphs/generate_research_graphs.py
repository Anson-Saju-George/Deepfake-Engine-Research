from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


BG = "#f6f3ee"
PANEL_BG = "#fffdf9"
GRID = "#d8d1c7"
TEXT = "#000000"
SUBTEXT = "#000000"
EDGE = "#d6cec2"
TRAIN_COLOR = "#a8b3c7"
VAL_COLOR = "#0f766e"
ACCENT = {
    "image": "#0f766e",
    "spatial": "#1d4ed8",
    "temporal": "#b45309",
    "spatiotemporal": "#9f1239",
    "unknown": "#475569",
}


@dataclass
class RunArtifact:
    run_dir: Path
    relative_dir: str
    run_name: str
    experiment_no: str
    protocol: str
    category: str
    family: str
    model_name: str
    dataset_scope: str
    dataset_names: list[str]
    loss_mode: str
    epochs_configured: int | None
    epochs_observed: int
    batch_size: int | None
    seq_len: int | None
    best_epoch: int | None
    best_val_f1: float | None
    best_val_acc: float | None
    test_loss: float | None
    test_acc: float | None
    test_f1: float | None
    train_samples: int | None
    val_samples: int | None
    test_samples: int | None
    test_fake_samples: int | None
    test_real_samples: int | None
    history: pd.DataFrame

    @property
    def category_color(self) -> str:
        return ACCENT.get(self.category, ACCENT["unknown"])

    @property
    def label(self) -> str:
        bits = [self.experiment_no, self.family, self.model_name]
        if self.protocol == "video":
            bits.append(self.loss_mode)
        return " | ".join(bit for bit in bits if bit and bit != "none")

    @property
    def slug(self) -> str:
        raw = f"{self.protocol}_{self.category}_{self.run_name}".lower()
        cleaned = []
        for char in raw:
            if char.isalnum():
                cleaned.append(char)
            elif char in {"-", "_"}:
                cleaned.append(char)
            else:
                cleaned.append("_")
        return "".join(cleaned).strip("_")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": EDGE,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Serif",
            "font.size": 14,
            "figure.titlesize": 22,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.frameon": False,
            "savefig.facecolor": BG,
            "savefig.bbox": "tight",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def wrap_label(value: str, width: int = 28) -> str:
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def prettify_token(value: str) -> str:
    return str(value).replace("_", " ").strip()


def make_model_label(experiment_no: str, family: str, model_name: str, loss_mode: str | None = None) -> str:
    top = prettify_token(model_name)
    bottom = prettify_token(loss_mode) if loss_mode and loss_mode != "none" else ""
    label = top if not bottom else f"{top}\n{bottom}"
    return wrap_label(label, width=34)


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_f1",
        "val_loss",
        "val_acc",
        "val_f1",
        "lr",
        "seconds",
    ]
    for column in expected:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def discover_runs(train_root: Path) -> list[RunArtifact]:
    runs: list[RunArtifact] = []
    for summary_path in sorted(train_root.rglob("final_summary.json")):
        run_dir = summary_path.parent
        history_path = run_dir / "history.csv"
        config_path = run_dir / "config.json"
        if not history_path.exists() or not config_path.exists():
            continue

        summary = read_json(summary_path)
        config = read_json(config_path)
        split = read_json(run_dir / "split_summary.json") if (run_dir / "split_summary.json").exists() else {}
        history = ensure_columns(pd.read_csv(history_path)).sort_values("epoch").reset_index(drop=True)

        relative_dir = run_dir.as_posix()
        protocol = "image" if "train/image/" in relative_dir else "video"
        category = str(summary.get("category") or config.get("category") or ("image" if protocol == "image" else "unknown")).lower()
        family = str(config.get("family") or config.get("family_dir") or run_dir.parent.name).replace("_", " ").strip()
        dataset_names = config.get("dataset_names") or summary.get("dataset_names") or []
        if not isinstance(dataset_names, list):
            dataset_names = [str(dataset_names)]

        metrics = summary.get("test_metrics", {})
        runs.append(
            RunArtifact(
                run_dir=run_dir,
                relative_dir=str(run_dir.relative_to(train_root.parent)).replace("\\", "/"),
                run_name=str(summary.get("run_name") or config.get("run_name") or run_dir.name),
                experiment_no=str(summary.get("experiment_no") or config.get("experiment_no") or run_dir.name),
                protocol=protocol,
                category=category,
                family=family,
                model_name=str(summary.get("model_name") or config.get("model_name") or "unknown"),
                dataset_scope=str(summary.get("dataset_scope") or config.get("dataset_scope") or "unknown"),
                dataset_names=[str(item) for item in dataset_names],
                loss_mode=str(summary.get("loss_mode") or config.get("loss_mode") or "none"),
                epochs_configured=safe_int(config.get("epochs")),
                epochs_observed=int(history["epoch"].max()) if not history.empty else 0,
                batch_size=safe_int(config.get("batch_size")),
                seq_len=safe_int(config.get("seq_len")),
                best_epoch=safe_int(summary.get("best_epoch")),
                best_val_f1=safe_float(summary.get("best_val_f1")),
                best_val_acc=safe_float(summary.get("best_val_acc")),
                test_loss=safe_float(metrics.get("loss")),
                test_acc=safe_float(metrics.get("acc")),
                test_f1=safe_float(metrics.get("f1")),
                train_samples=safe_int(split.get("train", {}).get("samples")) if isinstance(split, dict) else None,
                val_samples=safe_int(split.get("val", {}).get("samples")) if isinstance(split, dict) else None,
                test_samples=safe_int(split.get("test", {}).get("samples")) if isinstance(split, dict) else None,
                test_fake_samples=safe_int(split.get("test", {}).get("fake")) if isinstance(split, dict) else None,
                test_real_samples=safe_int(split.get("test", {}).get("real")) if isinstance(split, dict) else None,
                history=history,
            )
        )
    return runs


def add_epoch_marker(ax: plt.Axes, run: RunArtifact) -> None:
    if run.best_epoch is None:
        return
    ax.axvline(run.best_epoch, color=run.category_color, linewidth=1.2, linestyle="--", alpha=0.85)


def add_panel_header(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.tick_params(axis="x", rotation=0)


def plot_loss_panel(ax: plt.Axes, run: RunArtifact) -> None:
    frame = run.history
    ax.plot(frame["epoch"], frame["train_loss"], color=TRAIN_COLOR, linewidth=2.2, label="Train loss")
    ax.plot(frame["epoch"], frame["val_loss"], color=run.category_color, linewidth=2.6, label="Validation loss")
    add_epoch_marker(ax, run)
    add_panel_header(ax, "Loss Trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")


def plot_f1_panel(ax: plt.Axes, run: RunArtifact) -> None:
    frame = run.history
    ax.plot(frame["epoch"], frame["train_f1"], color=TRAIN_COLOR, linewidth=2.2, label="Train F1")
    ax.plot(frame["epoch"], frame["val_f1"], color=run.category_color, linewidth=2.8, label="Validation F1")
    if frame["val_f1"].notna().any():
        ax.fill_between(frame["epoch"], frame["val_f1"], color=run.category_color, alpha=0.12)
    add_epoch_marker(ax, run)
    add_panel_header(ax, "F1 Dynamics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, min(1.02, max(0.55, frame[["train_f1", "val_f1"]].max().max() + 0.03)))
    ax.legend(loc="lower right")


def plot_accuracy_panel(ax: plt.Axes, run: RunArtifact) -> None:
    frame = run.history
    ax.plot(frame["epoch"], frame["train_acc"], color=TRAIN_COLOR, linewidth=2.2, label="Train accuracy")
    ax.plot(frame["epoch"], frame["val_acc"], color=run.category_color, linewidth=2.8, label="Validation accuracy")
    add_epoch_marker(ax, run)
    add_panel_header(ax, "Accuracy Dynamics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, min(1.02, max(0.55, frame[["train_acc", "val_acc"]].max().max() + 0.03)))
    ax.legend(loc="lower right")


def plot_schedule_panel(ax: plt.Axes, run: RunArtifact) -> None:
    frame = run.history
    epochs = frame["epoch"].to_numpy()
    lr = frame["lr"].to_numpy(dtype=float)
    seconds = frame["seconds"].to_numpy(dtype=float)

    ax.plot(epochs, lr, color=run.category_color, linewidth=2.6, marker="o", markersize=4.5, label="Learning rate")
    add_epoch_marker(ax, run)
    add_panel_header(ax, "Schedule And Runtime")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate", color=run.category_color)
    ax.tick_params(axis="y", labelcolor=run.category_color)
    if np.isfinite(lr).all() and np.nanmax(lr) > 0:
        ax.set_yscale("log")

    twin = ax.twinx()
    twin.bar(epochs, seconds, color="#d9d2c4", alpha=0.45, width=0.65, label="Seconds")
    twin.set_ylabel("Seconds / epoch", color=SUBTEXT)
    twin.tick_params(axis="y", labelcolor=SUBTEXT)
    twin.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right")


def plot_summary_panel(ax: plt.Axes, run: RunArtifact) -> None:
    labels = ["Best val F1", "Test F1", "Best val acc", "Test acc"]
    values = [run.best_val_f1, run.test_f1, run.best_val_acc, run.test_acc]
    colors = [run.category_color, run.category_color, "#63748a", "#63748a"]
    y = np.arange(len(labels))
    numeric = [value if value is not None else 0.0 for value in values]

    ax.barh(y, numeric, color=colors, alpha=0.9, height=0.56)
    for idx, value in enumerate(values):
        ax.text((numeric[idx] if value is not None else 0.0) + 0.015, idx, format_metric(value), va="center", color=TEXT, fontsize=12.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    add_panel_header(ax, "Performance Scorecard")
    ax.set_xlabel("Score")

    dataset_str = ", ".join(run.dataset_names) if run.dataset_names else run.dataset_scope
    meta_lines = [
        f"Protocol: {run.protocol.title()} / {run.category.title()}",
        f"Family: {run.family}",
        f"Model: {run.model_name}",
        f"Loss: {run.loss_mode}",
        f"Epochs: {run.epochs_observed}/{run.epochs_configured or run.epochs_observed}",
        f"Batch size: {run.batch_size or 'n/a'}",
        f"Sequence length: {run.seq_len or 'n/a'}",
        f"Dataset: {dataset_str}",
        f"Split sizes: train {run.train_samples or 'n/a'} | val {run.val_samples or 'n/a'} | test {run.test_samples or 'n/a'}",
    ]
    ax.text(
        0.01,
        -0.22,
        "\n".join(meta_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11.2,
        color=SUBTEXT,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#faf7f1", "edgecolor": EDGE},
    )


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def add_figure_header(
    fig: plt.Figure,
    title: str,
    subtitle: str | None = None,
    footnote: str | None = None,
    *,
    title_x: float = 0.03,
    title_y: float = 0.992,
    subtitle_y: float = 0.945,
    footnote_y: float = 0.04,
) -> None:
    fig.suptitle(title, x=title_x, y=title_y, ha="left", fontweight="bold")
    if subtitle:
        fig.text(title_x, subtitle_y, subtitle, ha="left", va="top", fontsize=12.5, color=SUBTEXT, linespacing=1.35)
    if footnote:
        fig.text(title_x, footnote_y, footnote, ha="left", va="bottom", fontsize=10.8, color=SUBTEXT)


def add_run_figure_header(fig: plt.Figure, run: RunArtifact, title: str, note: str | None = None) -> None:
    subtitle = textwrap.fill(
        f"{prettify_token(run.model_name)}  |  "
        f"{run.protocol.title()} {run.category.title()}  |  loss = {prettify_token(run.loss_mode)}",
        width=92,
        break_long_words=False,
        break_on_hyphens=False,
    )
    add_figure_header(
        fig,
        title,
        subtitle=subtitle,
        footnote=(
            textwrap.fill(note, width=118, break_long_words=False, break_on_hyphens=False)
            if note
            else None
        ),
        title_x=0.02,
        title_y=0.994,
        subtitle_y=0.938,
        footnote_y=0.05,
    )


def make_run_figures(run: RunArtifact, out_dir: Path, dpi: int) -> list[Path]:
    base_dir = out_dir / run.protocol / run.category / run.slug
    generated: list[Path] = []

    specs = [
        ("loss_curve", "Loss Trajectory", plot_loss_panel, (11.8, 6.5), "Dashed marker shows the best validation epoch selected during training."),
        ("f1_curve", "F1 Dynamics", plot_f1_panel, (11.8, 6.5), "Dashed marker shows the best validation epoch selected during training."),
        ("accuracy_curve", "Accuracy Dynamics", plot_accuracy_panel, (11.8, 6.5), "Dashed marker shows the best validation epoch selected during training."),
        ("schedule_runtime", "Schedule And Runtime", plot_schedule_panel, (12.4, 6.8), "Learning rate is shown on a log scale when the schedule stays strictly positive."),
        ("scorecard", "Performance Scorecard", plot_summary_panel, (12.0, 7.6), None),
    ]

    for file_stem, title, plotter, figsize, note in specs:
        fig, ax = plt.subplots(figsize=figsize)
        plotter(ax, run)
        add_run_figure_header(fig, run, f"{prettify_token(run.model_name)} | {title}", note=note)
        if file_stem == "scorecard":
            fig.subplots_adjust(top=0.79, bottom=0.15)
        else:
            fig.subplots_adjust(top=0.79, bottom=0.12)
        out_base = base_dir / file_stem
        save_figure(fig, out_base, dpi=dpi)
        generated.append(out_base.with_suffix(".png"))

    return generated


def build_manifest_frame(runs: list[RunArtifact]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "run_name": run.run_name,
                "experiment_no": run.experiment_no,
                "protocol": run.protocol,
                "category": run.category,
                "family": run.family,
                "model_name": run.model_name,
                "loss_mode": run.loss_mode,
                "dataset_scope": run.dataset_scope,
                "best_epoch": run.best_epoch,
                "best_val_f1": run.best_val_f1,
                "best_val_acc": run.best_val_acc,
                "test_f1": run.test_f1,
                "test_acc": run.test_acc,
                "test_loss": run.test_loss,
                "epochs_observed": run.epochs_observed,
                "epochs_configured": run.epochs_configured,
                "batch_size": run.batch_size,
                "seq_len": run.seq_len,
                "train_samples": run.train_samples,
                "val_samples": run.val_samples,
                "test_samples": run.test_samples,
                "test_fake_samples": run.test_fake_samples,
                "test_real_samples": run.test_real_samples,
                "relative_dir": run.relative_dir,
            }
        )
    return pd.DataFrame(rows)


def classify_paradigm(run: RunArtifact) -> str:
    text = f"{run.family} {run.model_name}".lower()
    if "convnext" in text:
        return "CNN"
    if any(token in text for token in ["vit", "swin", "maxvit", "mvit", "transformer"]):
        return "Transformer"
    return "Other"


def select_best_run(runs: list[RunArtifact], protocol: str) -> RunArtifact | None:
    candidates = [run for run in runs if run.protocol == protocol and run.test_f1 is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda run: (run.test_f1 or -1.0, run.best_val_f1 or -1.0))


def load_prediction_frame(run: RunArtifact) -> pd.DataFrame | None:
    predictions_path = run.run_dir / "test_predictions.csv"
    if not predictions_path.exists():
        return None
    frame = pd.read_csv(predictions_path)
    required = {"label", "pred_label", "prob_real"}
    if not required.issubset(frame.columns):
        return None
    return frame


def derive_confusion_counts(run: RunArtifact) -> tuple[int, int, int, int] | None:
    if run.test_acc is None or run.test_f1 is None:
        return None
    if run.test_fake_samples is None or run.test_real_samples is None:
        return None

    # The stored F1 in these runs is aligned with the "real" class as positive.
    positives = int(run.test_real_samples)
    negatives = int(run.test_fake_samples)
    total = positives + negatives
    if total <= 0:
        return None

    error_target = (1.0 - run.test_acc) * total
    error_candidates = sorted({max(0, int(round(error_target))), max(0, int(math.floor(error_target))), max(0, int(math.ceil(error_target)))})
    best: tuple[float, tuple[int, int, int, int]] | None = None

    for errors in error_candidates:
        if run.test_f1 >= 0.999999:
            tp_est = positives
        else:
            tp_est = run.test_f1 * errors / max(1e-9, 2.0 * (1.0 - run.test_f1))
        center = int(round(tp_est))
        low = max(0, center - 400)
        high = min(positives, center + 400)
        for tp in range(low, high + 1):
            fn = positives - tp
            fp = errors - fn
            tn = negatives - fp
            if fp < 0 or tn < 0:
                continue
            acc = (tp + tn) / total
            denom = 2 * tp + fp + fn
            f1 = 0.0 if denom == 0 else (2 * tp) / denom
            score = abs(acc - run.test_acc) + abs(f1 - run.test_f1)
            if best is None or score < best[0]:
                best = (score, (tp, fn, fp, tn))

    return None if best is None else best[1]


def plot_confusion_matrix_figure(run: RunArtifact, out_dir: Path, dpi: int) -> None:
    prediction_frame = load_prediction_frame(run)
    note = "Held-out confusion matrix."
    if prediction_frame is not None:
        matrix = np.array(
            [
                [
                    int(((prediction_frame["label"] == 1) & (prediction_frame["pred_label"] == 1)).sum()),
                    int(((prediction_frame["label"] == 1) & (prediction_frame["pred_label"] == 0)).sum()),
                ],
                [
                    int(((prediction_frame["label"] == 0) & (prediction_frame["pred_label"] == 1)).sum()),
                    int(((prediction_frame["label"] == 0) & (prediction_frame["pred_label"] == 0)).sum()),
                ],
            ],
            dtype=float,
        )
        note = "Built from exported test predictions."
    else:
        counts = derive_confusion_counts(run)
        if counts is None:
            return
        tp, fn, fp, tn = counts
        matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(11.4, 7.9))
    image = ax.imshow(norm, cmap="coolwarm", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}\n{norm[i, j]:.2%}",
                ha="center",
                va="center",
                fontsize=15.8,
                color="#0f172a",
                fontweight="bold",
            )

    ax.set_xticks([0, 1], ["Predicted Real", "Predicted Fake"])
    ax.set_yticks([0, 1], ["Actual Real", "Actual Fake"])
    ax.set_title(f"{run.protocol.title()} Confusion Matrix", loc="left", pad=12)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(13.4)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
    colorbar.ax.tick_params(labelsize=12.6, colors=TEXT)
    colorbar.set_label("Row-normalized rate", color=TEXT, fontsize=13.2)
    add_run_figure_header(
        fig,
        run,
        f"{prettify_token(run.model_name)} | Confusion Matrix",
        note=note,
    )
    fig.subplots_adjust(left=0.12, right=0.92, top=0.8, bottom=0.14)

    save_figure(fig, out_dir / "overview" / f"{run.protocol}_confusion_matrix", dpi=dpi)


def plot_roc_curve_figure(run: RunArtifact, out_dir: Path, dpi: int) -> None:
    prediction_frame = load_prediction_frame(run)
    note = "Held-out ROC profile."
    if prediction_frame is not None and prediction_frame["label"].nunique() == 2:
        x, y, _ = roc_curve(prediction_frame["label"], prediction_frame["prob_real"], pos_label=1)
        auc = float(np.trapezoid(y, x))
        note = "Built from exported test probabilities."
    else:
        counts = derive_confusion_counts(run)
        if counts is None:
            return
        tp, fn, fp, tn = counts
        positives = max(1, tp + fn)
        negatives = max(1, tn + fp)
        tpr = tp / positives
        fpr = fp / negatives
        x = np.array([0.0, fpr, 1.0])
        y = np.array([0.0, tpr, 1.0])
        auc = float(np.trapezoid(y, x))

    fig, ax = plt.subplots(figsize=(11.8, 7.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa5b1", linewidth=1.4, label="Random baseline")
    ax.plot(x, y, color=run.category_color, linewidth=3.0, marker="o", markersize=6, label=f"ROC (AUC={auc:.3f})")
    ax.fill_between(x, y, alpha=0.18, color=run.category_color)
    if prediction_frame is not None and prediction_frame["label"].nunique() == 2:
        op_index = int(np.argmax(y - x))
        fpr = float(x[op_index])
        tpr = float(y[op_index])
        label_text = "Best Youden point"
    else:
        fpr = float(x[1])
        tpr = float(y[1])
        label_text = "Operating point"
    ax.scatter([fpr], [tpr], s=120, color="#7c2d12", edgecolors="white", linewidths=1.1, zorder=5)
    ax.annotate(
        f"{label_text}\nFPR={fpr:.3f}, TPR={tpr:.3f}",
        (fpr, tpr),
        xycoords="data",
        xytext=(0.97, 0.11),
        textcoords="axes fraction",
        fontsize=12,
        color=TEXT,
        linespacing=1.35,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#fffaf2", "edgecolor": EDGE, "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "color": "#7c2d12", "lw": 1.2, "shrinkA": 2, "shrinkB": 4},
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.margins(x=0.02, y=0.03)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{run.protocol.title()} ROC Profile", loc="left", pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.6, labelspacing=0.7, handlelength=2.2)
    add_run_figure_header(
        fig,
        run,
        f"{prettify_token(run.model_name)} | ROC Curve",
        note=note,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.14)

    save_figure(fig, out_dir / "overview" / f"{run.protocol}_roc_curve", dpi=dpi)


def make_leaderboard_figure(manifest: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    ranking = manifest.sort_values(["test_f1", "best_val_f1"], ascending=[False, False]).reset_index(drop=True)
    labels = [
        make_model_label(str(row.experiment_no), str(row.family), str(row.model_name), str(row.loss_mode))
        for row in ranking.itertuples(index=False)
    ]
    colors = [ACCENT.get(str(category), ACCENT["unknown"]) for category in ranking["category"]]
    y = np.arange(len(ranking))

    fig, axes = plt.subplots(1, 2, figsize=(18, max(7.5, len(ranking) * 0.48)), sharey=True)
    add_figure_header(
        fig,
        "Model Leaderboard Across All Discovered Runs",
        subtitle="Validation and held-out test performance, sorted by test F1. Colors encode model identity by training category.",
    )

    for ax, metric, title in zip(
        axes,
        ["best_val_f1", "test_f1"],
        ["Best Validation F1", "Held-out Test F1"],
        strict=True,
    ):
        values = ranking[metric].fillna(0.0).to_numpy()
        ax.barh(y, values, color=colors, alpha=0.92, height=0.66)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Score")
        ax.set_title(title, loc="left", pad=10)
        for idx, value in enumerate(values):
            ax.text(value + 0.012, idx, format_metric(ranking.iloc[idx][metric]), va="center", fontsize=11.8, color=TEXT)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(top=0.84, bottom=0.09, wspace=0.12)

    save_figure(fig, out_dir / "overview" / "leaderboard", dpi=dpi)


def make_generalization_figure(manifest: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    frame = manifest.dropna(subset=["best_val_f1", "test_f1"]).copy()
    if frame.empty:
        return

    frame["generalization_gap"] = frame["best_val_f1"] - frame["test_f1"]
    frame = frame.sort_values("generalization_gap", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(19.6, 8.4), gridspec_kw={"width_ratios": [1.0, 1.1]})
    add_figure_header(
        fig,
        "Generalization Profile",
        subtitle="The diagonal scatter highlights how closely each run's held-out test F1 tracks its best validation F1; the bar chart exposes the gap explicitly.",
    )

    scatter_ax, gap_ax = axes
    colors = [ACCENT.get(str(category), ACCENT["unknown"]) for category in frame["category"]]
    sizes = 160 + 260 * frame["test_acc"].fillna(frame["test_f1"]).clip(lower=0, upper=1).to_numpy()
    scatter_ax.scatter(frame["best_val_f1"], frame["test_f1"], s=sizes, c=colors, alpha=0.88, edgecolors="#ffffff", linewidths=1.2)
    line = np.linspace(0, 1, 100)
    scatter_ax.plot(line, line, linestyle="--", color="#7c8797", linewidth=1.5)
    scatter_ax.set_xlim(0, 1.02)
    scatter_ax.set_ylim(0, 1.02)
    scatter_ax.set_xlabel("Best validation F1")
    scatter_ax.set_ylabel("Held-out test F1")
    scatter_ax.set_title("Validation vs test", loc="left", pad=10)

    for row in frame.head(6).itertuples(index=False):
        label = prettify_token(row.model_name)
        if str(row.loss_mode) != "none":
            label = f"{label} | {prettify_token(row.loss_mode)}"
        scatter_ax.annotate(label, (row.best_val_f1, row.test_f1), xytext=(8, 6), textcoords="offset points", fontsize=10.8, color=TEXT)

    labels = [make_model_label(str(row.experiment_no), str(row.family), str(row.model_name), str(row.loss_mode)) for row in frame.itertuples(index=False)]
    gap_ax.barh(np.arange(len(frame)), frame["generalization_gap"], color=colors, alpha=0.9, height=0.65)
    gap_ax.set_yticks(np.arange(len(frame)), labels)
    gap_ax.invert_yaxis()
    gap_ax.set_xlabel("Best val F1 - test F1")
    gap_ax.set_title("Generalization gap", loc="left", pad=10)
    gap_ax.set_xlim(-0.03, 0.08)
    gap_ax.yaxis.tick_right()
    gap_ax.tick_params(axis="y", labelright=True, labelleft=False, pad=14)
    for tick in gap_ax.get_yticklabels():
        tick.set_fontsize(12.2)
        tick.set_linespacing(1.3)
    for idx, value in enumerate(frame["generalization_gap"]):
        if value >= 0:
            gap_ax.text(value + 0.008, idx, f"{value:.3f}", va="center", ha="left", fontsize=11, color=TEXT)
        else:
            gap_ax.text(value - 0.008, idx, f"{value:.3f}", va="center", ha="right", fontsize=11, color=TEXT)
    fig.subplots_adjust(top=0.84, bottom=0.1, wspace=0.18)

    save_figure(fig, out_dir / "overview" / "generalization_gap", dpi=dpi)


def make_category_comparisons(manifest: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    for category, group in manifest.groupby("category", sort=True):
        group = group.sort_values(["test_f1", "best_val_f1"], ascending=[False, False]).reset_index(drop=True)
        labels = [
            make_model_label(
                str(row.experiment_no),
                str(row.family),
                str(row.model_name),
                str(row.loss_mode),
            )
            for row in group.itertuples(index=False)
        ]
        label_line_counts = [label.count("\n") + 1 for label in labels]
        y = np.arange(len(group))
        color = ACCENT.get(str(category), ACCENT["unknown"])

        fig_height = max(7.3, 2.8 + sum(0.72 + 0.32 * (line_count - 1) for line_count in label_line_counts))
        fig, ax = plt.subplots(figsize=(17.2, fig_height))
        ax.barh(y - 0.18, group["best_val_f1"].fillna(0.0), height=0.34, color=color, alpha=0.92, label="Best val F1")
        ax.barh(y + 0.18, group["test_f1"].fillna(0.0), height=0.34, color="#9aa5b5", alpha=0.95, label="Test F1")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("F1 score")
        ax.set_title(f"{category.title()} Model Comparison", loc="left", pad=12)
        ax.legend(loc="upper left", bbox_to_anchor=(0.62, 1.02), ncol=2, columnspacing=1.2, handlelength=1.8)
        ax.tick_params(axis="y", pad=12)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(13)
            tick.set_linespacing(1.48)

        for idx, row in enumerate(group.itertuples(index=False)):
            ax.text((row.best_val_f1 or 0.0) + 0.01, idx - 0.18, format_metric(row.best_val_f1), va="center", fontsize=11.6)
            ax.text((row.test_f1 or 0.0) + 0.01, idx + 0.18, format_metric(row.test_f1), va="center", fontsize=11.6)

        subtitle = f"{len(group)} runs."
        fig.text(0.125, 0.936, subtitle, ha="left", va="top", fontsize=12.4, color=SUBTEXT)
        fig.text(0.125, 0.04, "Held-out test F1 is the primary ranking metric.", ha="left", va="bottom", fontsize=10.8, color=SUBTEXT)
        fig.subplots_adjust(left=0.39, right=0.97, top=0.875, bottom=0.09)
        save_figure(fig, out_dir / "overview" / f"{category}_comparison", dpi=dpi)


def make_cross_modality_comparison(manifest: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    summary = (
        manifest.groupby("category", as_index=False)
        .agg(
            best_test_f1=("test_f1", "max"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_acc=("test_acc", "mean"),
            run_count=("run_name", "count"),
        )
        .sort_values("best_test_f1", ascending=False)
    )
    if summary.empty:
        return

    colors = [ACCENT.get(str(category), ACCENT["unknown"]) for category in summary["category"]]
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.8))
    add_figure_header(
        fig,
        "Cross-Modality Comparison",
        subtitle="Held-out score summary by category.",
        footnote="Bars show category aggregates.",
    )

    x = np.arange(len(summary))
    width = 0.35
    axes[0].bar(x - width / 2, summary["best_test_f1"], width=width, color=colors, alpha=0.92, label="Best test F1")
    axes[0].bar(x + width / 2, summary["mean_test_f1"], width=width, color="#b4bdc9", alpha=0.98, label="Mean test F1")
    axes[0].set_xticks(x, [category.title() for category in summary["category"]])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("F1 score")
    axes[0].set_title("Best vs mean held-out F1", loc="left", pad=10)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.01, 1.01), ncol=2, columnspacing=1.1)
    for idx, row in enumerate(summary.itertuples(index=False)):
        axes[0].text(
            idx - width / 2,
            max(0.06, row.best_test_f1 - 0.055),
            f"{row.best_test_f1:.3f}",
            ha="center",
            va="top",
            fontsize=11.4,
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.92},
        )
        axes[0].text(
            idx + width / 2,
            max(0.06, row.mean_test_f1 - 0.055),
            f"{row.mean_test_f1:.3f}",
            ha="center",
            va="top",
            fontsize=11.4,
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.92},
        )

    axes[1].bar(x, summary["mean_test_acc"], width=0.58, color=colors, alpha=0.3, label="Mean test accuracy")
    axes[1].plot(x, summary["mean_test_f1"], color="#7c2d12", linewidth=2.6, marker="o", markersize=8, label="Mean test F1")
    axes[1].set_xticks(x, [category.title() for category in summary["category"]])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Average accuracy vs F1", loc="left", pad=10)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.01, 1.01))
    for idx, row in enumerate(summary.itertuples(index=False)):
        axes[1].text(idx, row.mean_test_acc + 0.015, f"n={row.run_count}", ha="center", fontsize=10.8, color=SUBTEXT)

    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.18)
    save_figure(fig, out_dir / "overview" / "cross_modality_comparison", dpi=dpi)


def make_cnn_vs_transformer_figure(manifest: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    frame = manifest.copy()
    frame["paradigm"] = [
        classify_paradigm(
            RunArtifact(
                run_dir=Path("."),
                relative_dir="",
                run_name=str(row.run_name),
                experiment_no=str(row.experiment_no),
                protocol=str(row.protocol),
                category=str(row.category),
                family=str(row.family),
                model_name=str(row.model_name),
                dataset_scope=str(row.dataset_scope),
                dataset_names=[],
                loss_mode=str(row.loss_mode),
                epochs_configured=None,
                epochs_observed=int(row.epochs_observed) if not pd.isna(row.epochs_observed) else 0,
                batch_size=None,
                seq_len=None,
                best_epoch=None,
                best_val_f1=None,
                best_val_acc=None,
                test_loss=None,
                test_acc=None,
                test_f1=None,
                train_samples=None,
                val_samples=None,
                test_samples=None,
                test_fake_samples=None,
                test_real_samples=None,
                history=pd.DataFrame(),
            )
        )
        for row in frame.itertuples(index=False)
    ]
    frame = frame[frame["paradigm"].isin(["CNN", "Transformer"])].copy()
    if frame.empty:
        return

    summary = (
        frame.groupby(["protocol", "paradigm"], as_index=False)
        .agg(mean_test_f1=("test_f1", "mean"), best_test_f1=("test_f1", "max"), run_count=("run_name", "count"))
        .sort_values(["protocol", "paradigm"])
    )
    protocols = ["image", "video"]
    paradigms = ["CNN", "Transformer"]
    x = np.arange(len(protocols))
    width = 0.32
    palette = {"CNN": "#0f766e", "Transformer": "#7c3aed"}

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.8))
    add_figure_header(
        fig,
        "CNN vs Transformer",
        subtitle="Architectural paradigm comparison.",
        footnote="Dots = individual runs.",
    )

    for ax, metric, title in zip(axes, ["mean_test_f1", "best_test_f1"], ["Mean held-out F1", "Best held-out F1"], strict=True):
        for offset, paradigm in zip([-width / 2, width / 2], paradigms, strict=True):
            values = []
            for protocol in protocols:
                row = summary[(summary["protocol"] == protocol) & (summary["paradigm"] == paradigm)]
                values.append(0.0 if row.empty else float(row.iloc[0][metric]))
            ax.bar(x + offset, values, width=width, color=palette[paradigm], alpha=0.9, label=paradigm if ax is axes[0] else None)

        for idx, protocol in enumerate(protocols):
            subset = frame[frame["protocol"] == protocol]
            for paradigm, x_anchor in zip(paradigms, [idx - width / 2, idx + width / 2], strict=True):
                points = subset[subset["paradigm"] == paradigm]["test_f1"].dropna().to_numpy()
                if len(points) == 0:
                    continue
                jitter = np.linspace(-0.045, 0.045, len(points))
                ax.scatter(np.full(len(points), x_anchor) + jitter, points, s=46, color="#f8fafc", edgecolors=palette[paradigm], linewidths=1.1, zorder=4)

        ax.set_xticks(x, [f"{protocol.title()} Models" for protocol in protocols])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("F1 score")
        ax.set_title(title, loc="left", pad=10)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, columnspacing=1.5, handlelength=1.8)
    fig.subplots_adjust(top=0.83, bottom=0.12, wspace=0.22)
    save_figure(fig, out_dir / "overview" / "cnn_vs_transformer", dpi=dpi)


def make_pipeline_diagram(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.0))
    ax.axis("off")
    add_figure_header(
        fig,
        "Deepfake Detection Pipeline",
        subtitle="Research workflow from raw datasets through protocol-separated training and final evaluation.",
    )

    boxes = [
        ((0.04, 0.60, 0.18, 0.18), "Raw Datasets", "CIFAKE\nAI-vs-Real Images\nCeleb-DF v2\nFaceForensics++", "#e0f2fe"),
        ((0.28, 0.60, 0.19, 0.18), "Cleaning And Standardization", "Corrupted sample removal\nResize / normalize\nFrame sampling rules", "#fef3c7"),
        ((0.53, 0.60, 0.18, 0.18), "Protocol Splitter", "Image-only protocol\nVideo-only protocol\nIdentity-aware partitions", "#dcfce7"),
        ((0.77, 0.67, 0.16, 0.11), "Image Track", "ConvNeXt / ViT / Swin", "#ede9fe"),
        ((0.77, 0.49, 0.16, 0.11), "Video Track", "Spatial / Temporal /\nSpatiotemporal", "#fee2e2"),
        ((0.30, 0.18, 0.19, 0.16), "Training Outputs", "history.csv\nfinal_summary.json\ncheckpoints", "#f3e8ff"),
        ((0.56, 0.18, 0.18, 0.16), "Evaluation Layer", "Accuracy\nF1\nConfusion matrix\nROC profile", "#fae8ff"),
        ((0.79, 0.18, 0.14, 0.16), "Paper Figures", "Leaderboards\nAblations\nCross-modality analysis", "#ede9fe"),
    ]

    for (x, y, w, h), title, body, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=EDGE, linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.015, y + h - 0.035, title, transform=ax.transAxes, fontsize=14.2, fontweight="bold", color=TEXT, va="top")
        ax.text(x + 0.015, y + h - 0.085, body, transform=ax.transAxes, fontsize=12.2, color=SUBTEXT, va="top", linespacing=1.35)

    arrows = [
        ((0.22, 0.69), (0.28, 0.69)),
        ((0.47, 0.69), (0.53, 0.69)),
        ((0.71, 0.71), (0.77, 0.725)),
        ((0.71, 0.63), (0.77, 0.545)),
        ((0.84, 0.67), (0.40, 0.34)),
        ((0.84, 0.49), (0.40, 0.34)),
        ((0.49, 0.26), (0.56, 0.26)),
        ((0.74, 0.26), (0.79, 0.26)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 2.0})

    fig.subplots_adjust(top=0.87, bottom=0.08)
    save_figure(fig, out_dir / "overview" / "pipeline_diagram", dpi=dpi)


def make_protocol_diagram(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    ax.axis("off")
    add_figure_header(
        fig,
        "Protocol-Separated Evaluation Design",
        subtitle="Image and video models are trained and evaluated under separate protocols to avoid modality leakage and unfair comparisons.",
    )

    lanes = [
        (0.08, "Image-Only Protocol", "#dcfce7", ["Image datasets only", "Preserve train/test source split", "Validation carved from train only", "Train image classifiers", "Evaluate on held-out image test set"]),
        (0.56, "Video-Only Protocol", "#fee2e2", ["Video datasets only", "Identity-aware 70/10/20 split", "Clip sampling fixed at eval", "Train spatial / temporal video models", "Evaluate on held-out video identities"]),
    ]
    for x, title, color, steps in lanes:
        ax.text(x, 0.84, title, transform=ax.transAxes, fontsize=15.5, fontweight="bold", color=TEXT, ha="left")
        y = 0.72
        for step in steps:
            rect = plt.Rectangle((x, y), 0.32, 0.10, facecolor=color, edgecolor=EDGE, linewidth=1.4, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(x + 0.015, y + 0.05, step, transform=ax.transAxes, fontsize=12.1, color=TEXT, va="center")
            if y > 0.30:
                ax.annotate("", xy=(x + 0.16, y - 0.01), xytext=(x + 0.16, y), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.9, "color": "#64748b"})
            y -= 0.13

    ax.text(0.5, 0.17, "Shared rule: compare results only after protocol separation, controlled preprocessing, and held-out testing.", transform=ax.transAxes, ha="center", fontsize=12.6, color=SUBTEXT, bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff7ed", "edgecolor": EDGE})
    fig.subplots_adjust(top=0.87, bottom=0.08)
    save_figure(fig, out_dir / "overview" / "protocol_diagram", dpi=dpi)


def make_loss_function_ablation(manifest: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    frame = manifest[
        (manifest["protocol"] == "video")
        & (manifest["model_name"] == "convnext_base")
        & (manifest["loss_mode"].isin(["none", "weighted_ce", "focal"]))
        & (manifest["category"].isin(["spatial", "temporal"]))
    ].copy()
    if frame.empty:
        return

    loss_order = ["none", "weighted_ce", "focal"]
    categories = ["spatial", "temporal"]
    x = np.arange(len(loss_order))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.3), sharey=True)
    add_figure_header(
        fig,
        "Loss Function Ablation",
        subtitle="ConvNeXt-Base video runs under matched settings. The chart compares best validation F1 and held-out test F1 across available loss formulations.",
    )

    for ax, category in zip(axes, categories, strict=True):
        subset = frame[frame["category"] == category].copy()
        subset["loss_mode"] = pd.Categorical(subset["loss_mode"], categories=loss_order, ordered=True)
        subset = subset.sort_values("loss_mode")
        val_map = {row.loss_mode: row.best_val_f1 for row in subset.itertuples(index=False)}
        test_map = {row.loss_mode: row.test_f1 for row in subset.itertuples(index=False)}
        val_values = [val_map.get(loss, np.nan) for loss in loss_order]
        test_values = [test_map.get(loss, np.nan) for loss in loss_order]

        ax.bar(x - width / 2, np.nan_to_num(val_values, nan=0.0), width=width, color=ACCENT.get(category, ACCENT["unknown"]), alpha=0.9, label="Best val F1")
        ax.bar(x + width / 2, np.nan_to_num(test_values, nan=0.0), width=width, color="#94a3b8", alpha=0.96, label="Test F1")
        ax.set_xticks(x, ["None", "Weighted CE", "Focal"])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{category.title()} video models", loc="left", pad=10)
        ax.set_ylabel("F1 score")
        for idx, value in enumerate(val_values):
            if not np.isnan(value):
                ax.text(idx - width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=11.6)
        for idx, value in enumerate(test_values):
            if not np.isnan(value):
                ax.text(idx + width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=11.6)
        missing = [loss for loss, value in zip(loss_order, test_values, strict=True) if np.isnan(value)]
        if missing:
            ax.text(0.02, 0.08, f"Unavailable in this repo: {', '.join(missing)}", transform=ax.transAxes, fontsize=10.8, color=SUBTEXT)

    axes[0].legend(loc="upper right")
    fig.subplots_adjust(top=0.84, bottom=0.1, wspace=0.15)
    save_figure(fig, out_dir / "overview" / "loss_function_ablation", dpi=dpi)


def write_readme(manifest: pd.DataFrame, out_dir: Path) -> None:
    category_counts = manifest.groupby("category").size().sort_index()
    best_row = manifest.sort_values(["test_f1", "best_val_f1"], ascending=[False, False]).iloc[0]
    lines = [
        "# Graph Outputs",
        "",
        "This folder is generated by `graphs/generate_research_graphs.py`.",
        "",
        f"- Total discovered runs: {len(manifest)}",
        f"- Best held-out test F1: {best_row['test_f1']:.3f} ({best_row['experiment_no']} | {best_row['family']} | {best_row['model_name']})",
        "- Output structure:",
        "  - `overview/`: cross-model comparison figures",
        "  - `runs/<protocol>/<category>/<run_slug>/`: per-run PNG figures",
        "  - `loss_curve.png`, `f1_curve.png`, `accuracy_curve.png`, `schedule_runtime.png`, `scorecard.png` for each run",
        "  - overview extras include confusion matrices, ROC profiles, cross-modality comparisons, protocol diagrams, and ablations",
        "  - `graph_manifest.csv`: machine-readable summary table",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in category_counts.items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Key Figures")
    lines.append("")
    lines.append("- `overview/leaderboard.png`")
    lines.append("- `overview/generalization_gap.png`")
    lines.append("- `overview/image_confusion_matrix.png`")
    lines.append("- `overview/video_confusion_matrix.png`")
    lines.append("- `overview/image_roc_curve.png`")
    lines.append("- `overview/video_roc_curve.png`")
    lines.append("- `overview/cross_modality_comparison.png`")
    lines.append("- `overview/cnn_vs_transformer.png`")
    lines.append("- `overview/pipeline_diagram.png`")
    lines.append("- `overview/protocol_diagram.png`")
    lines.append("- `overview/loss_function_ablation.png`")
    for category in category_counts.index:
        lines.append(f"- `overview/{category}_comparison.png`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Confusion matrices and ROC curves use exported `test_predictions.csv` when available.")
    lines.append("- If a run has not been re-evaluated yet, the graph generator falls back to reconstructed aggregate metrics from the saved summaries.")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality graphs for all discovered training runs.")
    parser.add_argument("--train-root", type=Path, default=Path("train"), help="Root directory containing training runs.")
    parser.add_argument("--output-dir", type=Path, default=Path("graphs"), help="Destination directory for generated figures.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    runs = discover_runs(args.train_root)
    if not runs:
        raise SystemExit("No training runs with history/config/final_summary were discovered.")

    run_out_dir = args.output_dir / "runs"
    generated = []
    for run in runs:
        generated.extend(make_run_figures(run, run_out_dir, dpi=args.dpi))

    manifest = build_manifest_frame(runs)
    manifest = manifest.sort_values(["protocol", "category", "family", "experiment_no", "run_name"]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "graph_manifest.csv", index=False)

    make_leaderboard_figure(manifest, args.output_dir, dpi=args.dpi)
    make_generalization_figure(manifest, args.output_dir, dpi=args.dpi)
    make_category_comparisons(manifest, args.output_dir, dpi=args.dpi)
    make_cross_modality_comparison(manifest, args.output_dir, dpi=args.dpi)
    make_cnn_vs_transformer_figure(manifest, args.output_dir, dpi=args.dpi)
    make_pipeline_diagram(args.output_dir, dpi=args.dpi)
    make_protocol_diagram(args.output_dir, dpi=args.dpi)
    make_loss_function_ablation(manifest, args.output_dir, dpi=args.dpi)
    best_image_run = select_best_run(runs, protocol="image")
    best_video_run = select_best_run(runs, protocol="video")
    if best_image_run is not None:
        plot_confusion_matrix_figure(best_image_run, args.output_dir, dpi=args.dpi)
        plot_roc_curve_figure(best_image_run, args.output_dir, dpi=args.dpi)
    if best_video_run is not None:
        plot_confusion_matrix_figure(best_video_run, args.output_dir, dpi=args.dpi)
        plot_roc_curve_figure(best_video_run, args.output_dir, dpi=args.dpi)
    write_readme(manifest, args.output_dir)

    print(f"Discovered runs: {len(runs)}")
    print(f"Per-run PNG figures: {len(generated)}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
