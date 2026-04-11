"""Video metadata inspection utility for real frame-count and duration analysis.

This script reads actual video metadata from files on disk to summarize frame
count, FPS, and duration distributions for each video dataset.

Interpretation for training:
- single-frame video loading is the spatial baseline
- contiguous sequence loading is the spatial+temporal path
- these stats describe raw-video mass, not a requirement to materialize all
  frames on disk
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from data.dataloader import VIDEO_EXTS
except ImportError:
    from dataloader import VIDEO_EXTS


def detect_dataset_name(path):
    parts = Path(path).parts
    if "videos" in parts:
        idx = parts.index("videos")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def get_label_from_path(path):
    path_str = str(path).lower()
    if "fake" in path_str or "\\ai\\" in path_str or "/ai/" in path_str:
        return 0
    if "real" in path_str:
        return 1
    return None


def get_video_stats(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "path": str(video_path),
            "opened": False,
            "frame_count": 0,
            "fps": 0.0,
            "duration_sec": 0.0,
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration_sec = (frame_count / fps) if fps > 0 else 0.0
    cap.release()

    return {
        "path": str(video_path),
        "opened": True,
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration_sec,
    }


def summarize_numeric(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)) if arr.size else 0.0,
        "p25": float(np.percentile(arr, 25)) if arr.size else 0.0,
        "median": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p75": float(np.percentile(arr, 75)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
    }


def print_summary(title, summary, unit=""):
    suffix = f" {unit}" if unit else ""
    print(f"\n{title}")
    print(f"  count : {summary['count']}")
    print(f"  min   : {summary['min']:.2f}{suffix}")
    print(f"  p25   : {summary['p25']:.2f}{suffix}")
    print(f"  median: {summary['median']:.2f}{suffix}")
    print(f"  p75   : {summary['p75']:.2f}{suffix}")
    print(f"  max   : {summary['max']:.2f}{suffix}")
    print(f"  mean  : {summary['mean']:.2f}{suffix}")


def print_mass_summary(title, items):
    if not items:
        return

    total_videos = len(items)
    total_frames = sum(item["frame_count"] for item in items)
    total_duration = sum(item["duration_sec"] for item in items)
    mean_frames = total_frames / total_videos if total_videos else 0.0
    mean_duration = total_duration / total_videos if total_videos else 0.0

    print(f"\n{title}")
    print(f"  videos              : {total_videos}")
    print(f"  total_frames        : {total_frames}")
    print(f"  total_duration_sec  : {total_duration:.2f}")
    print(f"  mean_frames_per_vid : {mean_frames:.2f}")
    print(f"  mean_duration_sec   : {mean_duration:.2f}")


def print_label_balance(dataset_name, label_groups):
    fake_items = label_groups.get("fake", [])
    real_items = label_groups.get("real", [])
    total_items = len(fake_items) + len(real_items)
    total_frames = sum(item["frame_count"] for item in fake_items + real_items)
    total_duration = sum(item["duration_sec"] for item in fake_items + real_items)

    if total_items == 0:
        return

    print("\nLabel balance")
    for label_name, label_items in [("fake", fake_items), ("real", real_items)]:
        if not label_items:
            continue

        label_videos = len(label_items)
        label_frames = sum(item["frame_count"] for item in label_items)
        label_duration = sum(item["duration_sec"] for item in label_items)

        video_share = (label_videos / total_items) * 100 if total_items else 0.0
        frame_share = (label_frames / total_frames) * 100 if total_frames else 0.0
        duration_share = (label_duration / total_duration) * 100 if total_duration else 0.0

        print(
            f"  {label_name:<4} | videos={label_videos:5d} ({video_share:6.2f}%) | "
            f"frames={label_frames:8d} ({frame_share:6.2f}%) | "
            f"duration={label_duration:9.2f}s ({duration_share:6.2f}%)"
        )

    if fake_items and real_items:
        fake_frames = sum(item["frame_count"] for item in fake_items)
        real_frames = sum(item["frame_count"] for item in real_items)
        dominant = "equal"
        if fake_frames > real_frames:
            dominant = "fake"
        elif real_frames > fake_frames:
            dominant = "real"

        print(f"  frame_mass_dominant : {dominant}")
        print(f"  dataset             : {dataset_name}")


def run(root="datasets", dataset_names=None, limit=None):
    root = Path(root)
    video_root = root / "videos"
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    allowed = set(dataset_names) if dataset_names else None

    videos = []
    for video_path in video_root.rglob("*"):
        if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTS:
            dataset_name = detect_dataset_name(video_path)
            if allowed and dataset_name not in allowed:
                continue
            videos.append(video_path)

    videos.sort()
    if limit is not None:
        videos = videos[:limit]

    print("\n=== VIDEO FRAME STATS ===")
    print(f"Root: {root}")
    print(f"Videos to inspect: {len(videos)}")
    if allowed:
        print(f"Dataset filter: {sorted(allowed)}")

    all_stats = []
    for video_path in tqdm(videos, desc="Inspecting videos"):
        all_stats.append(get_video_stats(video_path))

    failed = [item for item in all_stats if not item["opened"]]
    ok = [item for item in all_stats if item["opened"]]

    print(f"\nOpened successfully: {len(ok)}")
    print(f"Failed to open     : {len(failed)}")
    print("\nTraining interpretation:")
    print("- these frame counts describe raw video availability")
    print("- they do not imply that all frames should be materialized to disk")
    print("- sequence training samples contiguous clips on demand from raw videos")

    print_mass_summary("Global mass summary", ok)

    by_dataset = defaultdict(list)
    by_dataset_label = defaultdict(list)
    for item in ok:
        dataset_name = detect_dataset_name(item["path"])
        by_dataset[dataset_name].append(item)
        label = get_label_from_path(item["path"])
        by_dataset_label[(dataset_name, label)].append(item)

    for dataset_name in sorted(by_dataset):
        items = by_dataset[dataset_name]
        frame_counts = [item["frame_count"] for item in items]
        durations = [item["duration_sec"] for item in items]

        print(f"\n=== {dataset_name} ===")
        print_mass_summary("Dataset mass summary", items)
        print_summary("Frame count distribution", summarize_numeric(frame_counts), unit="frames")
        print_summary("Duration distribution", summarize_numeric(durations), unit="sec")

        label_groups = {
            "fake": by_dataset_label.get((dataset_name, 0), []),
            "real": by_dataset_label.get((dataset_name, 1), []),
        }
        print_label_balance(dataset_name, label_groups)
        for label_name, label_items in label_groups.items():
            if not label_items:
                continue
            label_frame_counts = [item["frame_count"] for item in label_items]
            print_summary(f"{label_name} frame count distribution", summarize_numeric(label_frame_counts), unit="frames")

    if failed:
        print("\n=== FAILED VIDEOS (first 20) ===")
        for item in failed[:20]:
            print(item["path"])


def main():
    parser = argparse.ArgumentParser(description="Inspect true frame counts and durations for video datasets.")
    parser.add_argument("--root", default="datasets", help="Dataset root path.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick checks.")
    args = parser.parse_args()

    run(root=args.root, dataset_names=args.datasets, limit=args.limit)


if __name__ == "__main__":
    main()
