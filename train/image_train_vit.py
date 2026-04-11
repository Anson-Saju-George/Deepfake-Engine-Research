"""ViT-only image trainer for IMG-EXP-01..03.

This is the first real trainer in the new top-level ``train/`` tree. It uses
the active protocol-aware dataloader and the finalized image-only split policy.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataloader import DatasetBuilder
from train.image_model import get_image_experiment_registry


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42
IMAGE_DATASET_TAGS = {
    "cifake": "cifake",
    "ai-generated-images-vs-real-images": "ai_gen",
    "image_combined": "image_combined",
}
VIT_EXPERIMENT_IDS = ["IMG-EXP-01", "IMG-EXP-02", "IMG-EXP-03"]
FAMILY_DIR_NAMES = {
    "ViT": "ViT",
    "ConvNeXt": "ConvNeXt",
    "EfficientNet": "EfficientNet",
    "ResNet": "ResNet",
}


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_scope_to_names(dataset_scope: str) -> list[str]:
    if dataset_scope == "cifake":
        return ["cifake"]
    if dataset_scope == "ai_gen":
        return ["ai-generated-images-vs-real-images"]
    if dataset_scope == "image_combined":
        return ["cifake", "ai-generated-images-vs-real-images"]
    raise ValueError(f"Unsupported dataset scope: {dataset_scope}")


def dataset_scope_to_tag(dataset_scope: str) -> str:
    if dataset_scope == "ai_gen":
        return IMAGE_DATASET_TAGS["ai-generated-images-vs-real-images"]
    return IMAGE_DATASET_TAGS[dataset_scope]


def build_vit_run_config(experiment_no: str, dataset_scope: str, batch_size: int | None = None) -> dict:
    registry = get_image_experiment_registry()
    if experiment_no not in VIT_EXPERIMENT_IDS:
        raise ValueError(f"{experiment_no} is not a ViT image experiment")

    config = deepcopy(registry[experiment_no])
    config["dataset_scope"] = dataset_scope
    config["dataset_names"] = dataset_scope_to_names(dataset_scope)
    config["dataset_tag"] = dataset_scope_to_tag(dataset_scope)
    if batch_size is not None:
        config["batch_size"] = batch_size

    run_name = f"{config['experiment_no']}_{config['model_name']}_{config['dataset_tag']}"
    family_dir = FAMILY_DIR_NAMES.get(config["family"], config["family"])
    save_dir = Path("train") / "image" / family_dir / run_name

    config["run_name"] = run_name
    config["family_dir"] = family_dir
    config["save_dir"] = str(save_dir)
    config["best_metric"] = "val_f1"
    config["save_threshold"] = 0.80
    config["seed"] = DEFAULT_SEED
    config["base_lr"] = 1e-4
    config["weight_decay"] = 1e-4
    config["min_lr"] = 1e-6
    config["warmup_epochs"] = 1
    config["grad_clip"] = 1.0
    config["label_smoothing"] = 0.1
    config["ema_decay"] = 0.999
    config["patience"] = 3
    config["min_delta"] = 1e-4
    return config


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


def accuracy_f1(preds: list[int], labels: list[int]) -> tuple[float, float]:
    if not labels:
        return 0.0, 0.0
    return accuracy_score(labels, preds), f1_score(labels, preds, zero_division=0)


def apply_warmup(optimizer, epoch: int, warmup_epochs: int, base_lr: float) -> None:
    if warmup_epochs <= 0:
        return
    if epoch > warmup_epochs:
        return
    scale = float(epoch) / float(warmup_epochs)
    warmup_lr = base_lr * scale
    for param_group in optimizer.param_groups:
        param_group["lr"] = warmup_lr


def format_metric(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
    ema=None,
    grad_clip=1.0,
    phase_name="run",
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    preds_all = []
    labels_all = []

    progress = tqdm(loader, leave=True, desc=phase_name, dynamic_ncols=True)
    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = DEVICE == "cuda"
        with autocast(device_type="cuda", enabled=autocast_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        preds_all.extend(preds.detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    acc, f1 = accuracy_f1(preds_all, labels_all)
    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": acc,
        "f1": f1,
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_history_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
                "lr",
                "seconds",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model_state: dict, payload: dict) -> None:
    torch.save(
        {
            "model_state_dict": model_state,
            **payload,
        },
        path,
    )


def summarize_split(loader) -> dict:
    samples = loader.dataset.samples
    label_counts = Counter(label for _, label, _ in samples)
    return {
        "samples": len(samples),
        "batches": len(loader),
        "fake": int(label_counts.get(0, 0)),
        "real": int(label_counts.get(1, 0)),
    }


def write_run_record(path: Path, config: dict, split_summary: dict, final_summary: dict | None = None) -> None:
    lines = [
        "# Run Record",
        "",
        "## Identity",
        "",
        f"- run_name: `{config['run_name']}`",
        f"- experiment_no: `{config['experiment_no']}`",
        f"- model_name: `{config['model_name']}`",
        f"- dataset_scope: `{config['dataset_scope']}`",
        f"- dataset_names: `{config['dataset_names']}`",
        f"- protocol: `{config['protocol']}`",
        f"- seed: `{config['seed']}`",
        "",
        "## Training Config",
        "",
        f"- epochs: `{config['epochs']}`",
        f"- batch_size: `{config['batch_size']}`",
        f"- balanced_sampling: `{config['balanced_sampling']}`",
        f"- base_lr: `{config['base_lr']}`",
        f"- min_lr: `{config['min_lr']}`",
        f"- weight_decay: `{config['weight_decay']}`",
        f"- ema_decay: `{config['ema_decay']}`",
        f"- label_smoothing: `{config['label_smoothing']}`",
        f"- patience: `{config['patience']}`",
        "",
        "## Split Summary",
        "",
    ]

    for split_name in ("train", "val", "test"):
        split = split_summary[split_name]
        lines.extend(
            [
                f"### {split_name.title()}",
                "",
                f"- samples: `{split['samples']}`",
                f"- batches: `{split['batches']}`",
                f"- fake: `{split['fake']}`",
                f"- real: `{split['real']}`",
                "",
            ]
        )

    if final_summary is not None:
        test_metrics = final_summary["test_metrics"]
        lines.extend(
            [
                "## Final Result",
                "",
                f"- best_epoch: `{final_summary['best_epoch']}`",
                f"- best_val_f1: `{format_metric(final_summary['best_val_f1'])}`",
                f"- best_val_acc: `{format_metric(final_summary['best_val_acc'])}`",
                f"- test_loss: `{test_metrics['loss']:.6f}`",
                f"- test_acc: `{test_metrics['acc']:.6f}`",
                f"- test_f1: `{test_metrics['f1']:.6f}`",
                f"- best_checkpoint: `{final_summary['best_checkpoint']}`",
                f"- last_checkpoint: `{final_summary['last_checkpoint']}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def train_vit_image_experiment(config: dict) -> dict:
    set_seed(config["seed"])

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "config.json"
    history_path = save_dir / "history.csv"
    best_summary_path = save_dir / "best_summary.json"
    final_summary_path = save_dir / "final_summary.json"
    split_summary_path = save_dir / "split_summary.json"
    run_record_path = save_dir / "run_record.md"
    best_path = save_dir / "best.pth"
    last_path = save_dir / "last.pth"

    save_json(config_path, config)

    print(f"Device: {DEVICE}")
    print(f"Run: {config['run_name']}")
    print(f"Save dir: {save_dir}")
    print(f"Datasets: {config['dataset_names']}")

    builder = DatasetBuilder(root="datasets", seed=config["seed"])
    train_loader, val_loader, test_loader = builder.get_loaders(
        batch_size=config["batch_size"],
        mode="single",
        dtype="image",
        protocol=config.get("protocol", "image_only"),
        dataset_names=config["dataset_names"],
        balanced=config.get("balanced_sampling", True),
    )

    split_summary = {
        "train": summarize_split(train_loader),
        "val": summarize_split(val_loader),
        "test": summarize_split(test_loader),
    }
    save_json(split_summary_path, split_summary)
    write_run_record(run_record_path, config, split_summary)

    model = timm.create_model(config["model_name"], pretrained=True, num_classes=2).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["base_lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["min_lr"],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    scaler = GradScaler(device="cuda", enabled=(DEVICE == "cuda"))
    ema = EMA(model, config["ema_decay"])

    best_f1 = float("-inf")
    best_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        print(f"\n===== EPOCH {epoch}/{config['epochs']} =====")
        apply_warmup(optimizer, epoch, config["warmup_epochs"], config["base_lr"])

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            grad_clip=config["grad_clip"],
            phase_name=f"train e{epoch}",
        )
        val_metrics = run_epoch(ema.ema, val_loader, criterion, phase_name=f"val e{epoch}")
        scheduler.step()
        epoch_seconds = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_acc": f"{train_metrics['acc']:.6f}",
            "train_f1": f"{train_metrics['f1']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_acc": f"{val_metrics['acc']:.6f}",
            "val_f1": f"{val_metrics['f1']:.6f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
            "seconds": f"{epoch_seconds:.2f}",
        }
        append_history_row(history_path, row)

        print(
            " | ".join(
                [
                    f"train_loss={train_metrics['loss']:.4f}",
                    f"train_acc={train_metrics['acc']:.4f}",
                    f"train_f1={train_metrics['f1']:.4f}",
                    f"val_loss={val_metrics['loss']:.4f}",
                    f"val_acc={val_metrics['acc']:.4f}",
                    f"val_f1={val_metrics['f1']:.4f}",
                    f"lr={optimizer.param_groups[0]['lr']:.7f}",
                    f"time={epoch_seconds:.1f}s",
                ]
            )
        )

        last_payload = {
            "epoch": epoch,
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(last_path, model.state_dict(), last_payload)

        improved = val_metrics["f1"] > (best_f1 + config["min_delta"])
        if improved:
            best_f1 = val_metrics["f1"]
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            patience_counter = 0

            best_payload = {
                "epoch": epoch,
                "config": config,
                "val_metrics": val_metrics,
            }
            save_checkpoint(best_path, ema.ema.state_dict(), best_payload)
            save_json(
                best_summary_path,
                {
                    "epoch": epoch,
                    "experiment_no": config["experiment_no"],
                    "model_name": config["model_name"],
                    "dataset_tag": config["dataset_tag"],
                    "val_f1": best_f1,
                    "val_acc": best_val_acc,
                    "threshold_promoted": best_val_acc >= config["save_threshold"],
                },
            )
            print("best.pth updated")
        else:
            patience_counter += 1
            print(f"no improvement | patience={patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print("early stopping triggered")
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, criterion, phase_name="test")
    print(
        f"\nTEST | loss={test_metrics['loss']:.4f} | "
        f"acc={test_metrics['acc']:.4f} | f1={test_metrics['f1']:.4f}"
    )

    final_summary = {
        "run_name": config["run_name"],
        "experiment_no": config["experiment_no"],
        "model_name": config["model_name"],
        "dataset_scope": config["dataset_scope"],
        "dataset_tag": config["dataset_tag"],
        "best_epoch": best_epoch,
        "best_val_f1": best_f1 if best_f1 != float("-inf") else None,
        "best_val_acc": best_val_acc,
        "test_metrics": test_metrics,
        "save_dir": str(save_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    save_json(final_summary_path, final_summary)
    write_run_record(run_record_path, config, split_summary, final_summary=final_summary)

    if best_val_acc >= config["save_threshold"] and best_path.exists():
        promoted_name = (
            f"final_{config['experiment_no']}_{config['model_name']}_{config['dataset_tag']}"
            f"_valf1{best_f1:.4f}_valacc{best_val_acc:.4f}.pth"
        )
        promoted_path = save_dir / promoted_name
        if not promoted_path.exists():
            promoted_path.write_bytes(best_path.read_bytes())

    return final_summary
