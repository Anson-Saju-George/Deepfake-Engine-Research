from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PREDICTIONS_CSV_NAME = "test_predictions.csv"
EVALUATION_JSON_NAME = "test_evaluation.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discover_completed_run_dirs(root: Path) -> list[Path]:
    runs = []
    for final_summary in sorted(root.rglob("final_summary.json")):
        run_dir = final_summary.parent
        if (run_dir / "config.json").exists():
            runs.append(run_dir)
    return runs


def resolve_checkpoint_path(run_dir: Path, checkpoint_preference: str) -> Path:
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    if checkpoint_preference == "best":
        return best_path
    if checkpoint_preference == "last":
        return last_path
    if best_path.exists():
        return best_path
    return last_path


def label_name(label: int) -> str:
    return "real" if int(label) == 1 else "fake"


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Prediction export received zero rows.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if np.isnan(result):
        return None
    return result


def build_evaluation_report(
    *,
    run_dir: Path,
    run_name: str,
    checkpoint_path: Path,
    protocol: str,
    mean_loss: float | None,
    labels: list[int],
    preds: list[int],
    prob_real: list[float],
    non_finite_score_count: int = 0,
    existing_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=int)
    preds_arr = np.asarray(preds, dtype=int)
    probs_arr = np.asarray(prob_real, dtype=float)

    metrics: dict[str, Any] = {
        "loss": _safe_float(mean_loss),
        "acc": float(accuracy_score(labels_arr, preds_arr)),
        "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "support_total": int(labels_arr.size),
        "support_real": int((labels_arr == 1).sum()),
        "support_fake": int((labels_arr == 0).sum()),
        "positive_label": 1,
        "positive_label_name": "real",
        "non_finite_score_count": int(non_finite_score_count),
    }

    finite_mask = np.isfinite(probs_arr)
    metrics["finite_score_count"] = int(finite_mask.sum())
    metrics["invalid_score_count"] = int((~finite_mask).sum())

    if len(np.unique(labels_arr)) == 2 and finite_mask.any():
        valid_labels = labels_arr[finite_mask]
        valid_probs = probs_arr[finite_mask]
        if len(np.unique(valid_labels)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(valid_labels, valid_probs))
            metrics["average_precision"] = float(average_precision_score(valid_labels, valid_probs))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    cm = confusion_matrix(labels_arr, preds_arr, labels=[1, 0])
    metrics["confusion_matrix"] = {
        "labels_order": ["real", "fake"],
        "matrix": cm.tolist(),
        "actual_real_pred_real": int(cm[0, 0]),
        "actual_real_pred_fake": int(cm[0, 1]),
        "actual_fake_pred_real": int(cm[1, 0]),
        "actual_fake_pred_fake": int(cm[1, 1]),
    }

    return {
        "run_name": run_name,
        "protocol": protocol,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "predictions_csv": str(run_dir / PREDICTIONS_CSV_NAME),
        "metrics": metrics,
        "source_final_summary": existing_summary or None,
    }
