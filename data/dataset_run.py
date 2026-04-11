"""Raw dataset readability validator for training-time safety checks.

This script enumerates every raw image and raw video sample discovered by
DatasetBuilder and attempts to read it with the same libraries used by the
project:

- PIL for images
- OpenCV for videos

Its purpose is to catch corrupted raw files before training begins. Frame-folder
validation is handled separately by `data.image_video_frame` because derived
frames are not part of the default training workflow.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

try:
    from data.dataloader import DatasetBuilder
except ImportError:
    from dataloader import DatasetBuilder


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False


def detect_dataset_name(path_str):
    parts = path_str.replace("\\", "/").split("/")
    if "images" in parts:
        idx = parts.index("images")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "videos" in parts:
        idx = parts.index("videos")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img = img.convert("RGB")
            _ = img.size

        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_video(path):
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False, "VideoOpenError: cv2.VideoCapture could not open file"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return False, "VideoMetadataError: invalid frame count (possible moov/metadata issue)"

        probe_indices = sorted(
            set(np.linspace(0, max(total_frames - 1, 0), num=min(3, total_frames), dtype=int))
        )
        successful_reads = 0

        for frame_idx in probe_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            successful_reads += 1

        if successful_reads == 0:
            return False, "VideoReadError: no readable frames"

        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()


def validate_sample(sample):
    path, label, sample_dtype = sample

    if sample_dtype == "image":
        ok, error = validate_image(path)
    elif sample_dtype == "video":
        ok, error = validate_video(path)
    else:
        ok, error = False, f"Unsupported dtype for dataset_run: {sample_dtype}"

    return path, label, sample_dtype, ok, error


def run_validation(root, dtype=None, limit=None, fail_fast=False, num_workers=8):
    builder = DatasetBuilder(root=root)
    builder.build()

    samples = builder.samples
    if dtype:
        samples = [sample for sample in samples if sample[2] == dtype]
    else:
        samples = [sample for sample in samples if sample[2] in {"image", "video"}]

    if limit is not None:
        samples = samples[:limit]

    print("\n=== DATASET RUN ===")
    print(f"Root: {root}")
    print(f"Filter dtype: {dtype or 'all'}")
    print(f"Samples to validate: {len(samples)}")
    print(f"Workers: {num_workers}")

    ok_counts = Counter()
    fail_counts = Counter()
    dataset_fail_counts = defaultdict(int)
    failures = []

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        iterator = executor.map(validate_sample, samples)
        for path, label, sample_dtype, ok, error in tqdm(iterator, total=len(samples), desc="Validating samples"):
            if ok:
                ok_counts[sample_dtype] += 1
                continue

            fail_counts[sample_dtype] += 1
            dataset_name = detect_dataset_name(path)
            dataset_fail_counts[dataset_name] += 1
            failures.append((path, label, sample_dtype, error))

            if fail_fast:
                executor.shutdown(wait=False, cancel_futures=True)
                break

    print("\n=== VALIDATION SUMMARY ===")
    total_ok = sum(ok_counts.values())
    total_fail = sum(fail_counts.values())
    print(f"Readable samples : {total_ok}")
    print(f"Failed samples   : {total_fail}")
    print(f"Readable by dtype: {dict(ok_counts)}")
    print(f"Failed by dtype  : {dict(fail_counts)}")

    if failures:
        print("\n=== FAILURES BY DATASET ===")
        for dataset_name in sorted(dataset_fail_counts):
            print(f"{dataset_name}: {dataset_fail_counts[dataset_name]}")

        print("\n=== FAILURE SAMPLES (first 50) ===")
        for path, label, sample_dtype, error in failures[:50]:
            print(f"[{sample_dtype}] label={label} | {path}")
            print(f"  error: {error}")
    else:
        print("\nAll dataloader-discovered files passed raw read validation.")

    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Validate every dataloader-discovered file by actually reading it."
    )
    parser.add_argument("--root", default="datasets", help="Dataset root path.")
    parser.add_argument("--dtype", choices=["image", "video"], default=None, help="Optional dtype filter.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick checks.")
    parser.add_argument("--num-workers", type=int, default=8, help="Thread count for parallel file validation.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first unreadable sample.")
    args = parser.parse_args()

    run_validation(
        root=args.root,
        dtype=args.dtype,
        limit=args.limit,
        fail_fast=args.fail_fast,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
