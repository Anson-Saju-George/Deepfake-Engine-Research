"""Root-level dataset distribution pull for training pipeline analysis.

This script reuses the active `DatasetBuilder` so the reported numbers match the
current training data pipeline. It can summarize the raw discovered corpus and
the protocol-aware split distribution that training would use.

Examples:
  python train_data_pipeline_pull.py
  python train_data_pipeline_pull.py --mode image
  python train_data_pipeline_pull.py --mode video
  python train_data_pipeline_pull.py --mode frame
  python train_data_pipeline_pull.py --mode video --datasets celeb-df-v2 faceforensics++
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from data.dataloader import DatasetBuilder, IMAGE_EXTS

LABEL_NAMES = {0: "fake", 1: "real"}


def percent(part, whole):
    if whole <= 0:
        return 0.0
    return 100.0 * float(part) / float(whole)


def count_saved_frames(frame_dir):
    frame_dir = Path(frame_dir)
    return sum(
        1 for item in frame_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def summarize_records(records):
    by_dtype = Counter()
    by_label = Counter()
    by_dataset = defaultdict(Counter)
    by_source_split = defaultdict(Counter)
    identity_counts = defaultdict(set)
    frame_mass = defaultdict(Counter)
    frame_lengths = defaultdict(list)

    for record in records:
        dtype = record["dtype"]
        dataset = record["dataset"]
        label = record["label"]
        source_split = record.get("source_split") or "unspecified"
        identity = record.get("identity")

        by_dtype[dtype] += 1
        by_label[label] += 1
        by_dataset[(dtype, dataset)][label] += 1
        by_source_split[(dtype, dataset)][source_split] += 1
        if identity is not None:
            identity_counts[(dtype, dataset)].add(identity)

        if dtype == "frame":
            frames = count_saved_frames(record["path"])
            frame_mass[dataset][label] += frames
            frame_lengths[dataset].append(frames)

    return {
        "by_dtype": by_dtype,
        "by_label": by_label,
        "by_dataset": by_dataset,
        "by_source_split": by_source_split,
        "identity_counts": identity_counts,
        "frame_mass": frame_mass,
        "frame_lengths": frame_lengths,
    }


def print_label_block(counts, prefix="  "):
    total = sum(counts.values())
    for label in sorted(counts):
        count = counts[label]
        print(f"{prefix}{LABEL_NAMES[label]:>4}: {count:7d} ({percent(count, total):6.2f}%)")


def print_raw_summary(records, summary):
    print("\n=== RAW DISCOVERY SUMMARY ===")
    print(f"Total records: {len(records)}")
    print(f"By dtype: {dict(summary['by_dtype'])}")
    print(
        "By label: "
        + ", ".join(
            f"{LABEL_NAMES[label]}={summary['by_label'][label]}"
            for label in sorted(summary["by_label"])
        )
    )

    for (dtype, dataset) in sorted(summary["by_dataset"]):
        counts = summary["by_dataset"][(dtype, dataset)]
        total = sum(counts.values())
        ids = len(summary["identity_counts"][(dtype, dataset)])
        print(f"\nDataset: {dataset} [{dtype}]")
        print(f"  records          : {total}")
        print(f"  unique identities: {ids}")
        print_label_block(counts)

        source_split_counts = summary["by_source_split"].get((dtype, dataset), {})
        if source_split_counts:
            split_line = ", ".join(
                f"{split_name}={count}"
                for split_name, count in sorted(source_split_counts.items())
            )
            print(f"  source splits    : {split_line}")

        if dtype == "frame":
            frame_mass = summary["frame_mass"].get(dataset, Counter())
            total_frames = sum(frame_mass.values())
            if total_frames > 0:
                print(f"  saved frame mass : {total_frames}")
                for label in sorted(frame_mass):
                    mass = frame_mass[label]
                    print(
                        f"    {LABEL_NAMES[label]:>4} frames: {mass:8d} ({percent(mass, total_frames):6.2f}%)"
                    )

            lengths = summary["frame_lengths"].get(dataset, [])
            if lengths:
                sorted_lengths = sorted(lengths)
                mid = len(sorted_lengths) // 2
                median = (
                    sorted_lengths[mid]
                    if len(sorted_lengths) % 2 == 1
                    else 0.5 * (sorted_lengths[mid - 1] + sorted_lengths[mid])
                )
                print(
                    "  frames/folder    : "
                    f"min={sorted_lengths[0]} "
                    f"median={median:.1f} "
                    f"max={sorted_lengths[-1]} "
                    f"mean={sum(sorted_lengths) / len(sorted_lengths):.2f}"
                )


def compute_protocol_split(builder, records, protocol, train_ratio=0.7, val_ratio=None):
    if val_ratio is None:
        val_ratio = 0.1 if protocol in {"video_only", "frame_only"} else 0.2

    if protocol == "image_only":
        train_records, val_records, test_records = builder._prepare_image_only(records)
    elif protocol == "video_only":
        train_records, val_records, test_records = builder._prepare_video_only(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    elif protocol == "frame_only":
        train_records, val_records, test_records = builder._prepare_frame_only(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    elif protocol == "combined_aux":
        train_records, val_records, test_records = builder._prepare_combined_aux(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "protocol": protocol,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }


def print_split_summary(split_data):
    protocol = split_data["protocol"]
    train_records = split_data["train"]
    val_records = split_data["val"]
    test_records = split_data["test"]
    total = len(train_records) + len(val_records) + len(test_records)

    print("\n=== TRAINING SPLIT SUMMARY ===")
    print(f"Protocol: {protocol}")
    print(
        f"Records : train={len(train_records)} val={len(val_records)} test={len(test_records)} total={total}"
    )

    for split_name, split_records in (
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ):
        print(f"\n[{split_name}]")
        print(f"  records: {len(split_records)} ({percent(len(split_records), total):6.2f}%)")

        label_counts = Counter(record["label"] for record in split_records)
        print_label_block(label_counts, prefix="  ")

        by_dataset = defaultdict(Counter)
        identities = defaultdict(set)
        frame_mass = defaultdict(Counter)

        for record in split_records:
            dataset = record["dataset"]
            label = record["label"]
            by_dataset[dataset][label] += 1
            identities[dataset].add(record.get("identity"))
            if record["dtype"] == "frame":
                frame_mass[dataset][label] += count_saved_frames(record["path"])

        for dataset in sorted(by_dataset):
            counts = by_dataset[dataset]
            dataset_total = sum(counts.values())
            print(
                f"  {dataset:30} | records={dataset_total:7d} | identities={len(identities[dataset]):6d}"
            )
            for label in sorted(counts):
                count = counts[label]
                print(
                    f"    {LABEL_NAMES[label]:>4}: {count:7d} ({percent(count, dataset_total):6.2f}%)"
                )

            if frame_mass.get(dataset):
                total_frames = sum(frame_mass[dataset].values())
                print(f"    frame mass total: {total_frames}")
                for label in sorted(frame_mass[dataset]):
                    mass = frame_mass[dataset][label]
                    print(
                        f"    {LABEL_NAMES[label]:>4} frame mass: {mass:8d} ({percent(mass, total_frames):6.2f}%)"
                    )


def resolve_mode(mode):
    if mode == "all":
        return None, "combined_aux"
    if mode == "image":
        return "image", "image_only"
    if mode == "video":
        return "video", "video_only"
    if mode == "frame":
        return "frame", "frame_only"
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Pull current training-pipeline dataset stats from the active dataloader."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "image", "video", "frame"],
        default="all",
        help="Media mode to analyze. Use frame later when extracted frame folders are ready.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filter, e.g. --datasets cifake or --datasets celeb-df-v2 faceforensics++",
    )
    parser.add_argument("--root", default="datasets", help="Dataset root path.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio for grouped protocols.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Optional validation ratio override. Defaults to protocol behavior.",
    )
    args = parser.parse_args()

    builder = DatasetBuilder(root=args.root)
    builder.build()

    dtype, protocol = resolve_mode(args.mode)
    records = builder._filter_records(dtype=dtype, dataset_names=args.datasets)

    if not records:
        print("No records matched the current selection.")
        return

    selection = f"mode={args.mode}"
    if args.datasets:
        selection += f" | datasets={args.datasets}"
    print(f"\n=== PIPELINE PULL ===\nSelection: {selection}")

    raw_summary = summarize_records(records)
    print_raw_summary(records, raw_summary)

    split_data = compute_protocol_split(
        builder,
        records,
        protocol=protocol,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print_split_summary(split_data)


if __name__ == "__main__":
    main()
