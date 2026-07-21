from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataloader import DeepFakeDataset, DatasetBuilder
from train.eval_predictions_common import (
    EVALUATION_JSON_NAME,
    PREDICTIONS_CSV_NAME,
    build_evaluation_report,
    discover_completed_run_dirs,
    label_name,
    load_json,
    resolve_checkpoint_path,
    save_json,
    write_predictions_csv,
)
from train.image.image_train import DEVICE, configure_runtime, set_seed


def default_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def resolve_loader_workers(workers: int | None) -> int:
    if workers is not None:
        return max(0, int(workers))
    return max(0, int(os.getenv("DF_NUM_WORKERS", "8")))


def make_loader_kwargs(*, batch_size: int, workers: int, prefetch_factor: int) -> dict:
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": (DEVICE == "cuda"),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def discover_image_runs() -> list[Path]:
    return discover_completed_run_dirs(Path("train") / "image")


def build_test_loader(
    config: dict,
    *,
    batch_size_override: int | None = None,
    workers: int | None = None,
    prefetch_factor: int = 4,
) -> tuple[DataLoader, list[dict]]:
    builder = DatasetBuilder(root="datasets", seed=int(config.get("seed", 42)))
    splits = builder.prepare_records(
        dtype="image",
        protocol=config.get("protocol", "image_only"),
        dataset_names=config["dataset_names"],
        balanced=bool(config.get("balanced_sampling", True)),
    )
    test_records = splits["test"]
    samples = [(record["path"], record["label"], record["dtype"]) for record in test_records]
    dataset = DeepFakeDataset(samples, transform=default_eval_transform(), mode="single", seq_len=1, clip_sampling="center")
    eval_batch_size = int(batch_size_override or config.get("batch_size", 32))
    loader = DataLoader(dataset, **make_loader_kwargs(batch_size=eval_batch_size, workers=resolve_loader_workers(workers), prefetch_factor=prefetch_factor))
    return loader, test_records


def load_image_model(config: dict, checkpoint_path: Path):
    model = timm.create_model(config["model_name"], pretrained=False, num_classes=2).to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_image_run(
    run_dir: Path,
    *,
    checkpoint_preference: str = "auto",
    batch_size: int | None = None,
    workers: int | None = None,
    prefetch_factor: int = 4,
    overwrite: bool = False,
) -> dict:
    predictions_path = run_dir / PREDICTIONS_CSV_NAME
    evaluation_path = run_dir / EVALUATION_JSON_NAME
    if not overwrite and predictions_path.exists() and evaluation_path.exists():
        return {"run_dir": str(run_dir), "status": "skipped_existing"}

    config = load_json(run_dir / "config.json")
    final_summary = load_json(run_dir / "final_summary.json")
    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint_preference)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for run: {run_dir}")

    set_seed(int(config.get("seed", 42)))
    configure_runtime()
    loader, test_records = build_test_loader(
        config,
        batch_size_override=batch_size,
        workers=workers,
        prefetch_factor=prefetch_factor,
    )
    model = load_image_model(config, checkpoint_path)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))

    rows = []
    labels = []
    preds = []
    prob_real = []
    total_loss = 0.0
    offset = 0
    non_finite_score_count = 0

    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc=f"eval {run_dir.name}", leave=True, dynamic_ncols=True):
            images = images.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)
            if images.ndim == 4 and DEVICE == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                outputs = model(images)
                safe_outputs = torch.nan_to_num(outputs.float(), nan=0.0, posinf=0.0, neginf=0.0)
                loss = criterion(safe_outputs, batch_labels)

            row_has_non_finite = ~torch.isfinite(outputs).all(dim=1)
            non_finite_score_count += int(row_has_non_finite.sum().item())
            probs = torch.softmax(safe_outputs, dim=1).detach().cpu()
            pred_labels = safe_outputs.argmax(dim=1).detach().cpu().tolist()
            label_list = batch_labels.detach().cpu().tolist()
            total_loss += float(loss.item())

            batch_size_now = len(label_list)
            batch_records = test_records[offset : offset + batch_size_now]
            offset += batch_size_now

            for local_idx, (record, true_label, pred_label) in enumerate(zip(batch_records, label_list, pred_labels, strict=True)):
                prob_fake = float(probs[local_idx, 0].item())
                prob_real_value = float(probs[local_idx, 1].item())
                labels.append(int(true_label))
                preds.append(int(pred_label))
                prob_real.append(prob_real_value)
                rows.append(
                    {
                        "sample_id": f"{run_dir.name}:{offset - batch_size_now + local_idx:06d}",
                        "path": record["path"],
                        "dataset_name": record.get("dataset", ""),
                        "identity": record.get("identity", ""),
                        "split": "test",
                        "run_name": config["run_name"],
                        "label": int(true_label),
                        "label_name": label_name(int(true_label)),
                        "pred_label": int(pred_label),
                        "pred_label_name": label_name(int(pred_label)),
                        "prob_fake": f"{prob_fake:.10f}",
                        "prob_real": f"{prob_real_value:.10f}",
                    }
                )

    write_predictions_csv(predictions_path, rows)
    report = build_evaluation_report(
        run_dir=run_dir,
        run_name=config["run_name"],
        checkpoint_path=checkpoint_path,
        protocol="image",
        mean_loss=(total_loss / max(len(loader), 1)),
        labels=labels,
        preds=preds,
        prob_real=prob_real,
        non_finite_score_count=non_finite_score_count,
        existing_summary=final_summary,
    )
    save_json(evaluation_path, report)
    return {
        "run_dir": str(run_dir),
        "status": "evaluated",
        "predictions_csv": str(predictions_path),
        "evaluation_json": str(evaluation_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export held-out test predictions for completed image runs.")
    parser.add_argument("--run-dir", action="append", default=[], help="Specific run directory to evaluate. Repeatable.")
    parser.add_argument("--checkpoint", choices=["auto", "best", "last"], default="auto", help="Checkpoint preference.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers. Defaults to DF_NUM_WORKERS or 8.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor when workers > 0.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction exports.")
    parser.add_argument("--list-runs", action="store_true", help="List discovered runs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [Path(path) for path in args.run_dir] if args.run_dir else discover_image_runs()
    runs = sorted(runs)

    if args.list_runs:
        for run in runs:
            print(run)
        print(f"TOTAL={len(runs)}")
        return

    if not runs:
        raise SystemExit("No completed image runs found.")

    evaluated = 0
    skipped = 0
    for run_dir in runs:
        result = evaluate_image_run(
            run_dir,
            checkpoint_preference=args.checkpoint,
            batch_size=args.batch_size,
            workers=args.workers,
            prefetch_factor=args.prefetch_factor,
            overwrite=args.overwrite,
        )
        print(f"{result['status']}: {result['run_dir']}")
        if result["status"] == "evaluated":
            evaluated += 1
        else:
            skipped += 1

    print(f"Image runs processed: total={len(runs)} evaluated={evaluated} skipped={skipped}")


if __name__ == "__main__":
    main()
