"""Runner for ViT image experiments.

Supports:
- CLI selection
- simple interactive menu when no args are provided
"""

from __future__ import annotations

import argparse

from train.image_train_vit import VIT_EXPERIMENT_IDS, build_vit_run_config, train_vit_image_experiment


DATASET_SCOPE_CHOICES = ["cifake", "ai_gen", "image_combined"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ViT image experiments from the new train tree.")
    parser.add_argument(
        "--exp",
        choices=VIT_EXPERIMENT_IDS,
        help="Experiment number to run.",
    )
    parser.add_argument(
        "--dataset-scope",
        choices=DATASET_SCOPE_CHOICES,
        help="Dataset scope to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override.",
    )
    return parser.parse_args()


def prompt_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(f"\n{prompt}")
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option}")
        choice = input("Select option: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid selection.")


def main():
    args = parse_args()

    exp = args.exp
    dataset_scope = args.dataset_scope
    batch_size = args.batch_size

    if exp is None:
        exp = prompt_choice("Choose ViT experiment", VIT_EXPERIMENT_IDS)
    if dataset_scope is None:
        dataset_scope = prompt_choice("Choose dataset scope", DATASET_SCOPE_CHOICES)

    config = build_vit_run_config(exp, dataset_scope, batch_size=batch_size)

    print("\nResolved run config")
    print(f"exp          : {config['experiment_no']}")
    print(f"model        : {config['model_name']}")
    print(f"dataset_scope: {config['dataset_scope']}")
    print(f"dataset_names: {config['dataset_names']}")
    print(f"epochs       : {config['epochs']}")
    print(f"batch_size   : {config['batch_size']}")
    print(f"save_dir     : {config['save_dir']}")

    train_vit_image_experiment(config)


if __name__ == "__main__":
    main()
