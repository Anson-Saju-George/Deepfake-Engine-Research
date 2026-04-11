"""Extract frame folders from videos discovered by the active dataloader.

This utility stays aligned with `data.dataloader.DatasetBuilder` instead of
maintaining a separate video discovery path. It is continuation-safe at the
video-folder level: completed folders are skipped, while partial or stale
folders are refreshed automatically on rerun.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from data.dataloader import DatasetBuilder
except ImportError:
    from dataloader import DatasetBuilder

DEFAULT_ROOT = "datasets"
DEFAULT_SAVE_ROOT = Path("datasets/preprocessed_frames")
MANIFEST_NAME = "extraction_meta.json"

def compute_indices(total_frames, frame_stride=1, max_frames=None):
    if total_frames <= 0:
        return []

    indices = list(range(0, total_frames, max(1, frame_stride)))
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        sampled = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
        indices = [indices[i] for i in sampled]
    return indices


def clear_existing_frames(save_dir):
    for frame_file in save_dir.glob("*.jpg"):
        frame_file.unlink()
    manifest_path = save_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()


def read_manifest(save_dir):
    manifest_path = save_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_manifest(save_dir, payload):
    manifest_path = save_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_frames(video_info):
    video_path, relative_path, save_root, frame_stride, max_frames, overwrite = video_info

    save_dir = Path(save_root) / Path(relative_path).with_suffix("")
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "failed_open", "path": str(video_path)}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = compute_indices(total_frames, frame_stride=frame_stride, max_frames=max_frames)

    if not indices:
        cap.release()
        return {"status": "failed_metadata", "path": str(video_path), "frame_count": total_frames}

    existing_frames = list(save_dir.glob("*.jpg"))
    manifest = read_manifest(save_dir)
    expected_count = len(indices)
    needs_refresh = False
    refresh_reason = None

    if len(existing_frames) == 16 and expected_count != 16:
        needs_refresh = True
        refresh_reason = "legacy_16_frame_output"

    if not overwrite and not needs_refresh and manifest:
        manifest_matches = (
            manifest.get("source_path") == str(video_path)
            and manifest.get("source_frame_count") == total_frames
            and manifest.get("frame_stride") == frame_stride
            and manifest.get("max_frames") == max_frames
            and manifest.get("saved_frame_count") == expected_count
        )
        if manifest_matches and len(existing_frames) == expected_count:
            cap.release()
            return {"status": "skipped", "path": str(video_path), "saved_frame_count": expected_count}

    if not overwrite and not needs_refresh:
        if existing_frames and len(existing_frames) != expected_count:
            needs_refresh = True
            refresh_reason = "partial_or_mismatched_frame_count"
        elif existing_frames and manifest is None:
            needs_refresh = True
            refresh_reason = "missing_manifest"
        elif existing_frames and manifest:
            manifest_matches = (
                manifest.get("source_path") == str(video_path)
                and manifest.get("source_frame_count") == total_frames
                and manifest.get("frame_stride") == frame_stride
                and manifest.get("max_frames") == max_frames
            )
            if not manifest_matches:
                needs_refresh = True
                refresh_reason = "stale_manifest"

    if overwrite or needs_refresh:
        clear_existing_frames(save_dir)

    saved_count = 0
    for out_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        save_path = save_dir / f"{out_idx:06d}.jpg"
        cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved_count += 1

    cap.release()

    write_manifest(
        save_dir,
        {
            "source_path": str(video_path),
            "source_frame_count": total_frames,
            "source_fps": fps,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "requested_indices": expected_count,
            "saved_frame_count": saved_count,
        },
    )

    status = "ok" if saved_count > 0 else "failed_decode"
    return {
        "status": status,
        "path": str(video_path),
        "source_frame_count": total_frames,
        "saved_frame_count": saved_count,
        "refresh_reason": refresh_reason,
    }


def build_video_tasks(root, save_root, dataset_names=None, frame_stride=1, max_frames=None, overwrite=False, limit=None):
    builder = DatasetBuilder(root=root)
    builder.build()

    allowed = set(dataset_names) if dataset_names else None
    tasks = []

    for record in builder.records:
        if record["dtype"] != "video":
            continue
        if allowed and record["dataset"] not in allowed:
            continue

        video_path = Path(record["path"])
        relative_path = video_path.relative_to(Path(root) / "videos")
        tasks.append(
            (
                str(video_path),
                str(relative_path),
                str(save_root),
                frame_stride,
                max_frames,
                overwrite,
            )
        )

    tasks.sort(key=lambda item: item[1])
    if limit is not None:
        tasks = tasks[:limit]
    return tasks

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos discovered by DatasetBuilder."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Dataset root passed to DatasetBuilder.")
    parser.add_argument("--save-root", default=str(DEFAULT_SAVE_ROOT), help="Output frame root.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Save every Nth frame.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on saved frames per video after stride is applied.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Process count.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if manifest matches.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of videos to process.")
    args = parser.parse_args()
    save_root = Path(args.save_root)

    print(
        f"Starting frame extraction | datasets={args.datasets or 'all'} | frame_stride={args.frame_stride} | "
        f"max_frames={args.max_frames or 'all'}"
    )
    video_tasks = build_video_tasks(
        root=args.root,
        save_root=save_root,
        dataset_names=args.datasets,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        limit=args.limit,
    )

    print(f"Videos discovered by dataloader: {len(video_tasks)}")
    if args.limit is not None:
        print(f"Video limit applied: {args.limit}")
    if args.datasets is None and args.frame_stride == 1 and args.max_frames is None:
        print("Warning: this is full-corpus extraction with every frame. It can take many hours and write a very large frame store.")
        print("Recommendation: start with --datasets celeb-df-v2 or --datasets faceforensics++ and/or use --max-frames 64.")

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(tqdm(executor.map(extract_frames, video_tasks), total=len(video_tasks)))

    ok_count = sum(1 for item in results if item["status"] == "ok")
    skipped_count = sum(1 for item in results if item["status"] == "skipped")
    failed = [item for item in results if item["status"] not in {"ok", "skipped"}]

    print(f"Finished. Extracted: {ok_count} | Skipped: {skipped_count} | Failed: {len(failed)}")
    print(f"Frames saved to: {save_root}")

    if failed:
        print("Sample failures:")
        for item in failed[:20]:
            print(f"{item['status']} | {item['path']}")


if __name__ == "__main__":
    main()

