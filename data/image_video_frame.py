"""Mixed-media dataset analyzer for images, videos, and extracted frame folders.

This module audits the current on-disk dataset state, compares it with
`DatasetBuilder` discovery, and can optionally validate raw readability or
smoke-test loader tensor outputs.

It is mainly useful when derived frame folders exist and need to be compared
against the raw image/video corpora. Raw-video training itself does not require
frame materialization.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

try:
    from data.dataloader import (
        DATASET_CONFIG,
        FRAME_DATASET_SOURCES,
        IMAGE_EXTS,
        VIDEO_EXTS,
        DatasetBuilder,
        DeepFakeDataset,
    )
except ImportError:
    from dataloader import (
        DATASET_CONFIG,
        FRAME_DATASET_SOURCES,
        IMAGE_EXTS,
        VIDEO_EXTS,
        DatasetBuilder,
        DeepFakeDataset,
    )

LABEL_NAMES = {0: "fake", 1: "real"}

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False


def infer_label_from_path(path):
    parts = [part.lower() for part in Path(path).parts]
    if "real" in parts or "youtube-real" in parts or "celeb-real" in parts:
        return 1
    if "fake" in parts or "celeb-fake" in parts or "ai" in parts:
        return 0

    path_str = str(path).lower().replace("\\", "/")
    if "/real/" in path_str:
        return 1
    if "/fake/" in path_str or "/ai/" in path_str:
        return 0
    return None


def is_frame_dir(path):
    path = Path(path)
    if not path.is_dir():
        return False
    return any(item.is_file() and item.suffix.lower() in IMAGE_EXTS for item in path.iterdir())


def count_frame_images(frame_dir):
    frame_dir = Path(frame_dir)
    return sum(
        1 for item in frame_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    )


def collect_real_stats(root):
    root = Path(root)
    stats = {
        "images": defaultdict(Counter),
        "videos": defaultdict(Counter),
        "frames": defaultdict(lambda: {"folders": Counter(), "frame_images": Counter()}),
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

    for source_name, source_cfg in FRAME_DATASET_SOURCES.items():
        source_root = source_cfg["root"]
        if not source_root.exists():
            continue

        if source_cfg["layout"] == "video_mirror":
            for dataset_dir in sorted([p for p in source_root.iterdir() if p.is_dir()]):
                dataset_name = dataset_dir.name
                for frame_dir in dataset_dir.rglob("*"):
                    if not is_frame_dir(frame_dir):
                        continue
                    label = infer_label_from_path(frame_dir)
                    if label is None:
                        continue
                    frame_count = count_frame_images(frame_dir)
                    stats["frames"][dataset_name]["folders"][(source_name, label)] += 1
                    stats["frames"][dataset_name]["frame_images"][(source_name, label)] += frame_count

        elif source_cfg["layout"] == "label_dataset_video":
            for label_dir in sorted([p for p in source_root.iterdir() if p.is_dir()]):
                label = infer_label_from_path(label_dir)
                if label is None:
                    continue
                for dataset_dir in sorted([p for p in label_dir.iterdir() if p.is_dir()]):
                    dataset_name = dataset_dir.name
                    for frame_dir in dataset_dir.iterdir():
                        if not is_frame_dir(frame_dir):
                            continue
                        frame_count = count_frame_images(frame_dir)
                        stats["frames"][dataset_name]["folders"][(source_name, label)] += 1
                        stats["frames"][dataset_name]["frame_images"][(source_name, label)] += frame_count

    return stats


def collect_loader_stats(root):
    builder = DatasetBuilder(root=root)
    builder.build()

    by_dtype = Counter()
    by_dataset = defaultdict(Counter)
    frame_mass = defaultdict(Counter)
    frame_len_stats = defaultdict(list)

    for record in builder.records:
        label = record["label"]
        dtype = record["dtype"]
        dataset_name = record["dataset"]
        by_dtype[dtype] += 1
        by_dataset[(dtype, dataset_name)][label] += 1

        if dtype == "frame":
            frame_count = count_frame_images(record["path"])
            frame_mass[dataset_name][label] += frame_count
            frame_len_stats[dataset_name].append(frame_count)

    return builder, by_dtype, by_dataset, frame_mass, frame_len_stats


def print_real_stats(stats):
    print("\n=== REAL MIXED-MEDIA STATS ===")

    print("\n[Images]")
    if not stats["images"]:
        print("No configured image datasets found.")
    else:
        for dataset_name in sorted(stats["images"]):
            split_counter = stats["images"][dataset_name]
            total = sum(split_counter.values())
            print(f"\nDataset: {dataset_name}")
            print(f"Total files: {total}")
            for split_name, label in sorted(split_counter):
                count = split_counter[(split_name, label)]
                print(f"  {split_name:>5} | {LABEL_NAMES[label]:>4}: {count}")

    print("\n[Videos]")
    if not stats["videos"]:
        print("No configured video datasets found.")
    else:
        for dataset_name in sorted(stats["videos"]):
            source_counter = stats["videos"][dataset_name]
            total = sum(source_counter.values())
            print(f"\nDataset: {dataset_name}")
            print(f"Total files: {total}")
            for source_name, label in sorted(source_counter):
                count = source_counter[(source_name, label)]
                print(f"  {source_name} | {LABEL_NAMES[label]}: {count}")

    print("\n[Frames]")
    if not stats["frames"]:
        print("No frame-folder datasets found.")
    else:
        for dataset_name in sorted(stats["frames"]):
            folder_counter = stats["frames"][dataset_name]["folders"]
            frame_counter = stats["frames"][dataset_name]["frame_images"]
            total_folders = sum(folder_counter.values())
            total_frames = sum(frame_counter.values())
            print(f"\nDataset: {dataset_name}")
            print(f"Total frame folders: {total_folders}")
            print(f"Total saved frames : {total_frames}")
            for source_name, label in sorted(folder_counter):
                folders = folder_counter[(source_name, label)]
                frames = frame_counter[(source_name, label)]
                print(
                    f"  {source_name:18} | {LABEL_NAMES[label]:>4} | "
                    f"folders={folders:6d} | frames={frames:8d}"
                )


def print_loader_stats(by_dtype, by_dataset, frame_mass, frame_len_stats):
    print("\n=== DATALOADER DISCOVERY STATS ===")
    print(f"By dtype: {dict(by_dtype)}")

    for (dtype, dataset_name) in sorted(by_dataset):
        counts = by_dataset[(dtype, dataset_name)]
        total = sum(counts.values())
        print(f"\nDataset: {dataset_name} [{dtype}]")
        print(f"Total discovered: {total}")
        for label in sorted(counts):
            print(f"  {LABEL_NAMES[label]}: {counts[label]}")

        if dtype == "frame":
            mass = frame_mass.get(dataset_name, Counter())
            if mass:
                total_frames = sum(mass.values())
                print(f"  frame folders mass: {total_frames}")
                for label in sorted(mass):
                    print(f"    {LABEL_NAMES[label]} frames: {mass[label]}")

                lengths = frame_len_stats.get(dataset_name, [])
                if lengths:
                    arr = torch.tensor(lengths, dtype=torch.float32)
                    print(
                        "  frames per folder: "
                        f"min={int(arr.min().item())} "
                        f"median={float(arr.median().item()):.1f} "
                        f"max={int(arr.max().item())} "
                        f"mean={float(arr.mean().item()):.2f}"
                    )
    print("\nTraining interpretation:")
    print("- image_only -> image-model spatial training")
    print("- video_only/single -> spatial baseline on raw videos")
    print("- video_only/sequence -> spatial+temporal raw-video training")
    print("- frame_only is auxiliary and only relevant when derived frame folders exist")


def compare_real_vs_loader(stats, by_dataset):
    print("\n=== REAL VS DATALOADER CHECK ===")
    real_totals = Counter()

    for dataset_name, split_counter in stats["images"].items():
        real_totals[("image", dataset_name)] += sum(split_counter.values())
    for dataset_name, source_counter in stats["videos"].items():
        real_totals[("video", dataset_name)] += sum(source_counter.values())
    for dataset_name, frame_stats in stats["frames"].items():
        real_totals[("frame", dataset_name)] += sum(frame_stats["folders"].values())

    discovered_totals = Counter()
    for key, counts in by_dataset.items():
        discovered_totals[key] += sum(counts.values())

    keys = sorted(set(real_totals) | set(discovered_totals))
    for dtype, dataset_name in keys:
        real_total = real_totals.get((dtype, dataset_name), 0)
        discovered_total = discovered_totals.get((dtype, dataset_name), 0)
        status = "OK" if real_total == discovered_total else "MISMATCH"
        print(
            f"{status:>8} | {dtype:>5} | {dataset_name:30} | "
            f"on_disk={real_total:7} | loader={discovered_total:7}"
        )


def validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_video(path):
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, "VideoOpenError"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return False, "VideoMetadataError"
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()


def validate_frame_dir(path):
    try:
        frame_files = sorted(
            item for item in Path(path).iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS
        )
        if not frame_files:
            return False, "FrameDirEmpty"

        probe_indices = sorted(set([0, len(frame_files) // 2, len(frame_files) - 1]))
        for idx in probe_indices:
            with Image.open(frame_files[idx]) as img:
                img.convert("RGB")
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_validation(builder, limit=None):
    print("\n=== RAW VALIDATION ===")
    records = builder.records[:limit] if limit is not None else builder.records
    results = Counter()
    failures = []

    for record in records:
        path = record["path"]
        dtype = record["dtype"]
        if dtype == "image":
            ok, error = validate_image(path)
        elif dtype == "video":
            ok, error = validate_video(path)
        elif dtype == "frame":
            ok, error = validate_frame_dir(path)
        else:
            ok, error = False, f"Unknown dtype: {dtype}"

        if ok:
            results[(dtype, "ok")] += 1
        else:
            results[(dtype, "fail")] += 1
            failures.append((dtype, path, error))

    for dtype in sorted({dtype for dtype, _ in results}):
        print(
            f"{dtype}: ok={results[(dtype, 'ok')]} fail={results[(dtype, 'fail')]}"
        )

    if failures:
        print("\nFirst failures:")
        for dtype, path, error in failures[:20]:
            print(f"[{dtype}] {path}")
            print(f"  error: {error}")


def run_smoke_loader(builder, seq_len=8, per_dtype=2):
    print("\n=== LOADER SMOKE TEST ===")
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    by_dtype = defaultdict(list)
    for record in builder.records:
        by_dtype[record["dtype"]].append((record["path"], record["label"], record["dtype"]))

    smoke_jobs = [
        ("image", "single"),
        ("video", "single"),
        ("video", "sequence"),
        ("frame", "single"),
        ("frame", "sequence"),
    ]

    for dtype, mode in smoke_jobs:
        samples = by_dtype.get(dtype, [])[:per_dtype]
        if not samples:
            print(f"{dtype}/{mode}: skipped (no samples)")
            continue

        dataset = DeepFakeDataset(samples, transform=tf, mode=mode, seq_len=seq_len, clip_sampling="center")
        try:
            sample, label = dataset[0]
            shape = tuple(sample.shape) if hasattr(sample, "shape") else "n/a"
            print(f"{dtype}/{mode}: ok | shape={shape} | label={label}")
        except Exception as exc:
            print(f"{dtype}/{mode}: failed | {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze images, videos, and extracted frame folders together."
    )
    parser.add_argument("--root", default="datasets", help="Dataset root path.")
    parser.add_argument("--validate", action="store_true", help="Read-check discovered records.")
    parser.add_argument("--smoke-loader", action="store_true", help="Smoke-test torch dataset loading.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for validation.")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length for smoke tests.")
    args = parser.parse_args()

    stats = collect_real_stats(args.root)
    builder, by_dtype, by_dataset, frame_mass, frame_len_stats = collect_loader_stats(args.root)

    print_real_stats(stats)
    print_loader_stats(by_dtype, by_dataset, frame_mass, frame_len_stats)
    compare_real_vs_loader(stats, by_dataset)

    if args.validate:
        run_validation(builder, limit=args.limit)

    if args.smoke_loader:
        run_smoke_loader(builder, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
