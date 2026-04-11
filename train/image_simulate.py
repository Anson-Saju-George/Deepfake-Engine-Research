"""Lightweight image training smoke test for the active data pipeline.

Purpose:
- verify that the image dataloader can build train/val/test splits correctly
- run one full training epoch end to end
- run validation and test passes after training
- exercise both image datasets separately by default

This is not a benchmark trainer. It is a pipeline-health check.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from data.dataloader import DatasetBuilder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASETS = [
    "cifake",
    "ai-generated-images-vs-real-images",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyImageSmokeNet(nn.Module):
    """Small CNN used only to validate the image pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


def summarize_loader(name: str, loader) -> None:
    dataset = loader.dataset
    label_counts = Counter(label for _, label, _ in dataset.samples)
    print(
        f"{name:<5} | samples={len(dataset):>7} | batches={len(loader):>5} | "
        f"fake={label_counts.get(0, 0):>7} | real={label_counts.get(1, 0):>7}"
    )


def run_epoch(model, loader, criterion, optimizer=None, phase_name="run"):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=True, desc=phase_name, dynamic_ncols=True):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    if not all_labels:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "f1": 0.0,
            "pred_dist": {},
            "label_dist": {},
        }

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "pred_dist": dict(Counter(int(x) for x in all_preds)),
        "label_dist": dict(Counter(int(x) for x in all_labels)),
    }


def run_simulation_for_dataset(dataset_name: str, batch_size: int) -> None:
    print("\n" + "=" * 80)
    print(f"IMAGE PIPELINE SMOKE TEST | dataset={dataset_name}")
    print("=" * 80)

    builder = DatasetBuilder(root="datasets", seed=42)
    train_loader, val_loader, test_loader = builder.get_loaders(
        batch_size=batch_size,
        mode="single",
        dtype="image",
        protocol="image_only",
        dataset_names=[dataset_name],
        balanced=True,
    )

    print("Split summary")
    summarize_loader("train", train_loader)
    summarize_loader("val", val_loader)
    summarize_loader("test", test_loader)

    model = TinyImageSmokeNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nTraining for 1 epoch")
    train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer, phase_name="train")

    print("Running validation")
    val_metrics = run_epoch(model, val_loader, criterion, phase_name="val")

    print("Running test")
    test_metrics = run_epoch(model, test_loader, criterion, phase_name="test")

    print("\nResults")
    for split_name, metrics in (
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ):
        print(
            f"{split_name:<5} | loss={metrics['loss']:.4f} | "
            f"acc={metrics['acc']:.4f} | f1={metrics['f1']:.4f} | "
            f"pred={metrics['pred_dist']} | labels={metrics['label_dist']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run a 1-epoch image pipeline smoke test.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Image datasets to test. Defaults to both main image datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for the smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    print(f"Device: {DEVICE}")
    print(f"Datasets: {args.datasets}")
    print(f"Batch size: {args.batch_size}")

    for dataset_name in args.datasets:
        run_simulation_for_dataset(dataset_name, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
