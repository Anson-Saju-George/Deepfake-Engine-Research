"""Unified experiment-driven image training runner."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from train.image.image_models import get_image_experiment_registry
from train.image.image_train import (
    DEFAULT_SEED,
    FAMILY_DIR_NAMES,
    IMAGE_DATASET_TAGS,
    dataset_scope_to_names,
    dataset_scope_to_tag,
    train_image_experiment,
    with_image_train_log,
)

DATASET_SCOPE_CHOICES = ["cifake", "ai_gen", "image_combined"]


def build_image_run_config(experiment_no: str, dataset_scope: str, batch_size: int | None = None) -> dict:
    registry = get_image_experiment_registry()
    if experiment_no not in registry:
        raise ValueError(f"Unknown image experiment: {experiment_no}")
    if dataset_scope not in DATASET_SCOPE_CHOICES:
        raise ValueError(f"Unsupported dataset scope: {dataset_scope}")

    config = deepcopy(registry[experiment_no])
    config["dataset_scope"] = dataset_scope
    config["dataset_names"] = dataset_scope_to_names(dataset_scope)
    config["dataset_tag"] = dataset_scope_to_tag(dataset_scope)
    if batch_size is not None:
        config["batch_size"] = batch_size

    config["run_name"] = f"{config['experiment_no']}_{config['model_name']}_{config['dataset_tag']}"
    config["family_dir"] = FAMILY_DIR_NAMES.get(config["family"], config["family"])
    config["save_dir"] = str(Path("train") / "image" / config["family_dir"] / config["run_name"])
    config.update({
        "best_metric": "val_f1",
        "save_threshold": 0.80,
        "seed": DEFAULT_SEED,
        "base_lr": 1e-4,
        "weight_decay": 1e-4,
        "min_lr": 1e-6,
        "warmup_epochs": 1,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "ema_decay": 0.999,
        "patience": 3,
        "min_delta": 1e-4,
    })
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run any configured image experiment.")
    parser.add_argument("--exp", choices=sorted(get_image_experiment_registry()), help="Experiment ID.")
    parser.add_argument("--dataset-scope", choices=DATASET_SCOPE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args(argv)


def prompt_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(f"\n{prompt}")
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option}")
        choice = input("Select option: ").strip()
        if choice.isdigit() and 0 <= int(choice) - 1 < len(options):
            return options[int(choice) - 1]
        print("Invalid selection.")


def main(argv=None) -> None:
    args = parse_args(argv)
    exp = args.exp or prompt_choice("Choose image experiment", sorted(get_image_experiment_registry()))
    scope = args.dataset_scope or prompt_choice("Choose dataset scope", DATASET_SCOPE_CHOICES)
    config = build_image_run_config(exp, scope, batch_size=args.batch_size)
    with with_image_train_log(config):
        print("\nResolved run config")
        for key in ("experiment_no", "family", "model_name", "dataset_scope", "dataset_names", "epochs", "batch_size", "base_lr", "save_dir"):
            print(f"{key:<13}: {config[key]}")
        train_image_experiment(config)


if __name__ == "__main__":
    main()