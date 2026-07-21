from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
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
from train.video.video_train_backbone import (
    DEVICE,
    TimmVideoClassifier,
    build_criterion,
    compute_class_weights_from_samples,
    configure_runtime,
    set_seed,
)


def discover_video_runs() -> list[Path]:
    return discover_completed_run_dirs(Path("train") / "video")


def default_video_eval_transform(config: dict):
    runtime = config.get("runtime", {})
    image_size = int(runtime.get("image_size", 224))
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def resolve_video_workers(config: dict, workers: int | None) -> int:
    if workers is not None:
        return max(0, int(workers))
    runtime = config.get("runtime", {})
    return max(0, int(runtime.get("workers", os.getenv("DF_NUM_WORKERS", "8"))))


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


def configure_video_runtime(config: dict) -> None:
    runtime = config["runtime"]
    os.environ["DF_VIDEO_DECODE_BACKEND"] = runtime["decode_backend"]
    os.environ["DF_NUM_WORKERS"] = str(runtime["workers"])
    os.environ["DF_PREFETCH_FACTOR"] = str(runtime["prefetch_factor"])
    os.environ["DF_VIDEO_READER_CACHE"] = str(runtime["reader_cache"])
    os.environ.pop("DF_FFMPEG_OUTPUT_SIZE", None)


def build_video_loaders(
    config: dict,
    *,
    batch_size_override: int | None = None,
    workers: int | None = None,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, list[tuple[str, int, str]], list[dict]]:
    builder = DatasetBuilder(root="datasets", seed=int(config.get("seed", 42)))
    splits = builder.prepare_records(
        dtype="video",
        protocol="video_only",
        dataset_names=config["dataset_names"],
        balanced=(config["loss_mode"] == "none"),
    )
    train_samples = [(record["path"], record["label"], record["dtype"]) for record in splits["train"]]
    test_records = splits["test"]
    test_samples = [(record["path"], record["label"], record["dtype"]) for record in test_records]

    batch_size_value = int(batch_size_override or config.get("batch_size", 1))
    worker_count = resolve_video_workers(config, workers)
    resolved_prefetch = int(prefetch_factor if prefetch_factor is not None else config.get("runtime", {}).get("prefetch_factor", 4))
    test_dataset = DeepFakeDataset(
        test_samples,
        transform=default_video_eval_transform(config),
        mode=config["mode"],
        seq_len=int(config.get("seq_len", 1)),
        clip_sampling="center",
    )
    test_loader = DataLoader(test_dataset, **make_loader_kwargs(batch_size=batch_size_value, workers=worker_count, prefetch_factor=resolved_prefetch))
    return test_loader, train_samples, test_records


def load_video_model(config: dict, checkpoint_path: Path):
    model = TimmVideoClassifier(config["model_name"]).to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_video_run(
    run_dir: Path,
    *,
    checkpoint_preference: str = "auto",
    batch_size: int | None = None,
    workers: int | None = None,
    prefetch_factor: int | None = None,
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
    configure_video_runtime(config)
    test_loader, train_samples, test_records = build_video_loaders(
        config,
        batch_size_override=batch_size,
        workers=workers,
        prefetch_factor=prefetch_factor,
    )
    model = load_video_model(config, checkpoint_path)
    class_weights = compute_class_weights_from_samples(train_samples)
    criterion, _ = build_criterion(
        config["loss_mode"],
        class_weights,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )

    rows = []
    labels = []
    preds = []
    prob_real = []
    total_loss = 0.0
    offset = 0
    non_finite_score_count = 0

    with torch.inference_mode():
        for inputs, batch_labels in tqdm(test_loader, desc=f"eval {run_dir.name}", leave=True, dynamic_ncols=True):
            inputs = inputs.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)
            if inputs.ndim == 4 and DEVICE == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                outputs = model(inputs)
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
        protocol="video",
        mean_loss=(total_loss / max(len(test_loader), 1)),
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
    parser = argparse.ArgumentParser(description="Export held-out test predictions for completed video runs.")
    parser.add_argument("--run-dir", action="append", default=[], help="Specific run directory to evaluate. Repeatable.")
    parser.add_argument("--checkpoint", choices=["auto", "best", "last"], default="auto", help="Checkpoint preference.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers. Defaults to run runtime workers.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch factor. Defaults to run runtime prefetch.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction exports.")
    parser.add_argument("--list-runs", action="store_true", help="List discovered runs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [Path(path) for path in args.run_dir] if args.run_dir else discover_video_runs()
    runs = sorted(runs)

    if args.list_runs:
        for run in runs:
            print(run)
        print(f"TOTAL={len(runs)}")
        return

    if not runs:
        raise SystemExit("No completed video runs found.")

    evaluated = 0
    skipped = 0
    for run_dir in runs:
        result = evaluate_video_run(
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

    print(f"Video runs processed: total={len(runs)} evaluated={evaluated} skipped={skipped}")


if __name__ == "__main__":
    main()
