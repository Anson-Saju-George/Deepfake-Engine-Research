from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.image.test_image_models import discover_image_runs, evaluate_image_run
from train.video.test_video_models import discover_video_runs, evaluate_video_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export held-out predictions for all completed image and video runs.")
    parser.add_argument("--checkpoint", choices=["auto", "best", "last"], default="auto", help="Checkpoint preference.")
    parser.add_argument("--batch-size-image", type=int, default=None, help="Optional image batch size override.")
    parser.add_argument("--batch-size-video", type=int, default=None, help="Optional video batch size override.")
    parser.add_argument("--workers-image", type=int, default=None, help="Image DataLoader workers.")
    parser.add_argument("--workers-video", type=int, default=None, help="Video DataLoader workers.")
    parser.add_argument("--prefetch-image", type=int, default=4, help="Image DataLoader prefetch factor when workers > 0.")
    parser.add_argument("--prefetch-video", type=int, default=None, help="Video DataLoader prefetch factor when workers > 0.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction exports.")
    parser.add_argument("--image-only", action="store_true", help="Only process image runs.")
    parser.add_argument("--video-only", action="store_true", help="Only process video runs.")
    parser.add_argument("--list-runs", action="store_true", help="List discovered runs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    do_images = not args.video_only
    do_videos = not args.image_only
    image_runs = discover_image_runs() if do_images else []
    video_runs = discover_video_runs() if do_videos else []

    if args.list_runs:
        for run in image_runs:
            print(f"image\t{run}")
        for run in video_runs:
            print(f"video\t{run}")
        print(f"TOTAL={len(image_runs) + len(video_runs)}")
        return

    total = 0
    evaluated = 0
    skipped = 0

    for run_dir in image_runs:
        total += 1
        result = evaluate_image_run(
            run_dir,
            checkpoint_preference=args.checkpoint,
            batch_size=args.batch_size_image,
            workers=args.workers_image,
            prefetch_factor=args.prefetch_image,
            overwrite=args.overwrite,
        )
        print(f"image | {result['status']}: {result['run_dir']}")
        if result["status"] == "evaluated":
            evaluated += 1
        else:
            skipped += 1

    for run_dir in video_runs:
        total += 1
        result = evaluate_video_run(
            run_dir,
            checkpoint_preference=args.checkpoint,
            batch_size=args.batch_size_video,
            workers=args.workers_video,
            prefetch_factor=args.prefetch_video,
            overwrite=args.overwrite,
        )
        print(f"video | {result['status']}: {result['run_dir']}")
        if result["status"] == "evaluated":
            evaluated += 1
        else:
            skipped += 1

    print(f"Completed evaluation export: total={total} evaluated={evaluated} skipped={skipped}")


if __name__ == "__main__":
    main()
