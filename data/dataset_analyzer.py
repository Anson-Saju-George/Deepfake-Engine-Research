"""Dataset audit script for comparing real filesystem counts with loader discovery.

Use this module to verify that the files present on disk match what the active
DatasetBuilder discovers. It is intended for dataset accounting, coverage checks,
and paper-ready reporting before training.

This script focuses on the raw corpora used by the main training path:
- image datasets for image-only experiments
- raw video datasets for video-only experiments

Derived frame folders are intentionally left to `data.image_video_frame`, since
frame materialization is auxiliary rather than the default workflow.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

try:
    from data.dataloader import DATASET_CONFIG, IMAGE_EXTS, VIDEO_EXTS, DatasetBuilder
except ImportError:
    from dataloader import DATASET_CONFIG, IMAGE_EXTS, VIDEO_EXTS, DatasetBuilder

LABEL_NAMES = {0: "fake", 1: "real"}


def detect_dataset_name(path_str):
    path = Path(path_str)
    parts = path.parts

    if "images" in parts:
        idx = parts.index("images")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if "videos" in parts:
        idx = parts.index("videos")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    return "unknown"


def collect_real_stats(root):
    root = Path(root)
    stats = {
        "images": defaultdict(Counter),
        "videos": defaultdict(Counter),
    }

    image_root = root / "images"
    if image_root.exists():
        for dataset_name, cfg in DATASET_CONFIG["images"].items():
            ds_path = image_root / dataset_name
            if not ds_path.exists():
                continue

            for split_dir in sorted([p for p in ds_path.iterdir() if p.is_dir()]):
                for class_name, label in cfg["classes"].items():
                    cls_dir = split_dir / class_name
                    if not cls_dir.exists():
                        continue

                    count = sum(
                        1
                        for item in cls_dir.rglob("*")
                        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
                    )
                    if count:
                        stats["images"][dataset_name][(split_dir.name, label)] += count

    video_root = root / "videos"
    if video_root.exists():
        for dataset_name, cfg in DATASET_CONFIG["videos"].items():
            ds_path = video_root / dataset_name
            if not ds_path.exists():
                continue

            if cfg["structure"] == "flat_class":
                for class_name, label in cfg["classes"].items():
                    cls_dir = ds_path / class_name
                    if not cls_dir.exists():
                        continue

                    count = sum(
                        1
                        for item in cls_dir.rglob("*")
                        if item.is_file() and item.suffix.lower() in VIDEO_EXTS
                    )
                    if count:
                        stats["videos"][dataset_name][(class_name, label)] += count

            elif cfg["structure"] == "deep_nested":
                for rel_path in cfg["rules"]["fake_paths"]:
                    nested_dir = ds_path / rel_path
                    if not nested_dir.exists():
                        continue

                    count = sum(
                        1
                        for item in nested_dir.rglob("*")
                        if item.is_file() and item.suffix.lower() in VIDEO_EXTS
                    )
                    if count:
                        stats["videos"][dataset_name][(rel_path, 0)] += count

                for rel_path in cfg["rules"]["real_paths"]:
                    nested_dir = ds_path / rel_path
                    if not nested_dir.exists():
                        continue

                    count = sum(
                        1
                        for item in nested_dir.rglob("*")
                        if item.is_file() and item.suffix.lower() in VIDEO_EXTS
                    )
                    if count:
                        stats["videos"][dataset_name][(rel_path, 1)] += count

    return stats


def collect_loader_stats(root):
    builder = DatasetBuilder(root=root)
    builder.build()

    by_dataset = defaultdict(Counter)
    by_dtype = Counter()
    by_label = Counter()

    for path, label, dtype in builder.samples:
        if dtype not in {"image", "video"}:
            continue
        dataset_name = detect_dataset_name(path)
        by_dataset[dataset_name][label] += 1
        by_dtype[dtype] += 1
        by_label[label] += 1

    raw_samples = [sample for sample in builder.samples if sample[2] in {"image", "video"}]
    return raw_samples, by_dataset, by_dtype, by_label


def print_real_stats(real_stats):
    print("\n=== REAL DATASET STATS (FILES ON DISK) ===")

    print("\n[Images]")
    if not real_stats["images"]:
        print("No configured image datasets found.")
    else:
        for dataset_name in sorted(real_stats["images"]):
            split_counter = real_stats["images"][dataset_name]
            total = sum(split_counter.values())
            print(f"\nDataset: {dataset_name}")
            print(f"Total files: {total}")
            for split_name, label in sorted(split_counter):
                count = split_counter[(split_name, label)]
                print(f"  {split_name:>5} | {LABEL_NAMES[label]:>4}: {count}")

    print("\n[Videos]")
    if not real_stats["videos"]:
        print("No configured video datasets found.")
    else:
        for dataset_name in sorted(real_stats["videos"]):
            source_counter = real_stats["videos"][dataset_name]
            total = sum(source_counter.values())
            print(f"\nDataset: {dataset_name}")
            print(f"Total files: {total}")
            for source_name, label in sorted(source_counter):
                count = source_counter[(source_name, label)]
                print(f"  {source_name} | {LABEL_NAMES[label]}: {count}")


def print_loader_stats(samples, by_dataset, by_dtype, by_label):
    print("\n=== DATALOADER DISCOVERY STATS ===")
    print(f"Total raw samples discovered: {len(samples)}")
    print(f"By dtype: {dict(by_dtype)}")
    print(
        "By label: "
        + ", ".join(f"{LABEL_NAMES[label]}={count}" for label, count in sorted(by_label.items()))
    )

    for dataset_name in sorted(by_dataset):
        counts = by_dataset[dataset_name]
        total = sum(counts.values())
        print(f"\nDataset: {dataset_name}")
        print(f"Total discovered: {total}")
        for label in sorted(counts):
            print(f"  {LABEL_NAMES[label]}: {counts[label]}")

    print("\nTraining interpretation:")
    print("- image datasets feed image-only spatial models")
    print("- raw video datasets feed video-only models")
    print("- video single mode is the spatial video baseline")
    print("- video sequence mode is the spatial+temporal path")


def compare_real_vs_loader(real_stats, by_dataset):
    print("\n=== REAL VS DATALOADER CHECK ===")
    mismatches = []

    real_totals = Counter()
    for dataset_name, split_counter in real_stats["images"].items():
        real_totals[dataset_name] += sum(split_counter.values())
    for dataset_name, source_counter in real_stats["videos"].items():
        real_totals[dataset_name] += sum(source_counter.values())

    discovered_totals = Counter()
    for dataset_name, counts in by_dataset.items():
        discovered_totals[dataset_name] += sum(counts.values())

    dataset_names = sorted(set(real_totals) | set(discovered_totals))
    for dataset_name in dataset_names:
        real_total = real_totals.get(dataset_name, 0)
        discovered_total = discovered_totals.get(dataset_name, 0)
        status = "OK" if real_total == discovered_total else "MISMATCH"
        print(
            f"{status:>8} | {dataset_name:30} | on_disk={real_total:7} | loader={discovered_total:7}"
        )
        if real_total != discovered_total:
            mismatches.append(dataset_name)

    if not mismatches:
        print("\nLoader discovery matches configured on-disk raw dataset stats.")
    else:
        print("\nDatasets with mismatches:")
        for dataset_name in mismatches:
            print(f"  - {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze real dataset stats and compare them with raw dataloader discovery."
    )
    parser.add_argument("--root", default="datasets", help="Dataset root path.")
    args = parser.parse_args()

    real_stats = collect_real_stats(args.root)
    samples, by_dataset, by_dtype, by_label = collect_loader_stats(args.root)

    print_real_stats(real_stats)
    print_loader_stats(samples, by_dataset, by_dtype, by_label)
    compare_real_vs_loader(real_stats, by_dataset)


if __name__ == "__main__":
    main()
