"""Unified command-line runner for all active video experiments."""

from __future__ import annotations

import argparse

from train.video.video_models import CATEGORY_GUIDE, get_video_experiment_registry
from train.video.video_train_backbone import (
    VIDEO_DATASET_SCOPE_CHOICES,
    build_video_run_config,
    normalize_loss_mode,
    prompt_choice,
    run_video_experiment_smoke,
    with_video_train_log,
)


def _experiment_ids(category: str | None = None) -> list[str]:
    registry = get_video_experiment_registry()
    return [
        experiment_id
        for experiment_id, config in registry.items()
        if category is None or config["category"] == category
    ]


def parse_args(argv=None, *, category: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configured raw-video experiment from the unified registry."
    )
    parser.add_argument("--exp", choices=_experiment_ids(category), help="Experiment ID.")
    parser.add_argument(
        "--category",
        choices=[category] if category else list(CATEGORY_GUIDE),
        default=category,
        help="Optional category filter when using the unified command.",
    )
    parser.add_argument("--dataset-scope", choices=VIDEO_DATASET_SCOPE_CHOICES)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--loss-mode",
        choices=["none", "weighted_bce", "weighted_ce", "focal", "focal_loss"],
        default="none",
    )
    return parser.parse_args(argv)


def _print_config(config: dict) -> None:
    print("\nResolved run config")
    for key in (
        "experiment_no", "family", "model_name", "category", "dataset_scope",
        "dataset_names", "protocol", "mode", "seq_len", "epochs", "batch_size",
        "base_lr", "loss_mode", "temporal_head", "save_dir",
    ):
        if key in config:
            print(f"{key:<13}: {config[key]}")


def main(argv=None, *, category: str | None = None) -> None:
    args = parse_args(argv, category=category)
    experiment_ids = _experiment_ids(args.category or category)
    exp = args.exp or prompt_choice("Choose video experiment", experiment_ids)
    dataset_scope = args.dataset_scope or prompt_choice(
        "Choose dataset scope", VIDEO_DATASET_SCOPE_CHOICES
    )
    config = build_video_run_config(
        exp,
        dataset_scope,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        epochs=args.epochs,
        loss_mode=normalize_loss_mode(args.loss_mode),
        base_lr=args.lr,
    )
    if args.category and config["category"] != args.category:
        raise SystemExit(
            f"Experiment {exp} belongs to category '{config['category']}', "
            f"not '{args.category}'."
        )

    with with_video_train_log(config):
        _print_config(config)
        run_video_experiment_smoke(config)


if __name__ == "__main__":
    main()
