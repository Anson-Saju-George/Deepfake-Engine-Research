"""Extract frame folders using FFmpeg hardware acceleration.

This mirrors the continuation-safe behavior of `proc.pre_process_videos`, but
delegates frame extraction to FFmpeg and targets Intel Quick Sync (`qsv`) by
default. The name is kept for compatibility with the previous experimental path.

Important:
- this is a best-effort hardware-accelerated extractor
- codec support depends on the installed FFmpeg build and the source codec
- unsupported videos can optionally fall back to CPU FFmpeg decode
"""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data.dataloader import DatasetBuilder
except ImportError:
    from dataloader import DatasetBuilder

DEFAULT_ROOT = "datasets"
DEFAULT_SAVE_ROOT = Path("temp/hw_preprocessed_frames")
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


def probe_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return {"total_frames": total_frames, "fps": fps}


def build_select_expression(indices):
    if not indices:
        return None
    return "+".join(f"eq(n\\,{int(idx)})" for idx in indices)


def build_ffmpeg_command(video_path, output_pattern, indices, jpeg_quality, hwaccel, decoder):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]

    if hwaccel:
        command.extend(["-hwaccel", hwaccel])

    if decoder and decoder != "auto":
        command.extend(["-c:v", decoder])

    command.extend(["-i", str(video_path)])

    vf_parts = []
    if indices:
        vf_parts.append(f"select='{build_select_expression(indices)}'")
        vf_parts.append("setpts=N/FRAME_RATE/TB")

    if vf_parts:
        command.extend(["-vf", ",".join(vf_parts)])

    command.extend(
        [
            "-vsync",
            "0",
            "-q:v",
            str(jpeg_quality),
            str(output_pattern),
        ]
    )
    return command


def run_ffmpeg_extract(video_path, save_dir, indices, jpeg_quality, hwaccel, decoder):
    output_pattern = save_dir / "%06d.jpg"
    command = build_ffmpeg_command(
        video_path=video_path,
        output_pattern=output_pattern,
        indices=indices,
        jpeg_quality=jpeg_quality,
        hwaccel=hwaccel,
        decoder=decoder,
    )
    return subprocess.run(command, capture_output=True, text=True)


def count_saved_frames(save_dir):
    return sum(1 for _ in save_dir.glob("*.jpg"))


def extract_frames(video_info):
    (
        video_path,
        relative_path,
        save_root,
        frame_stride,
        max_frames,
        overwrite,
        hwaccel,
        decoder,
        allow_cpu_fallback,
        jpeg_quality,
    ) = video_info

    save_dir = Path(save_root) / Path(relative_path).with_suffix("")
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata = probe_video_metadata(video_path)
    if metadata is None:
        return {"status": "failed_open", "path": str(video_path)}

    total_frames = int(metadata["total_frames"])
    fps = float(metadata["fps"])
    indices = compute_indices(total_frames, frame_stride=frame_stride, max_frames=max_frames)

    if not indices:
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
            return {
                "status": "skipped",
                "path": str(video_path),
                "saved_frame_count": expected_count,
            }

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

    result = run_ffmpeg_extract(
        video_path=video_path,
        save_dir=save_dir,
        indices=indices,
        jpeg_quality=jpeg_quality,
        hwaccel=hwaccel,
        decoder=decoder,
    )

    backend_used = f"ffmpeg_{hwaccel or 'cpu'}"
    ffmpeg_error = None
    if result.returncode != 0:
        ffmpeg_error = (result.stderr or result.stdout or "").strip()
        if allow_cpu_fallback:
            clear_existing_frames(save_dir)
            fallback = run_ffmpeg_extract(
                video_path=video_path,
                save_dir=save_dir,
                indices=indices,
                jpeg_quality=jpeg_quality,
                hwaccel=None,
                decoder=None,
            )
            result = fallback
            backend_used = "ffmpeg_cpu_fallback"
            if result.returncode != 0:
                ffmpeg_error = (result.stderr or result.stdout or "").strip()

    saved_count = count_saved_frames(save_dir)

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
            "backend": backend_used,
            "decoder": decoder,
            "hwaccel": hwaccel,
        },
    )

    if result.returncode != 0 or saved_count <= 0:
        return {
            "status": "failed_decode",
            "path": str(video_path),
            "source_frame_count": total_frames,
            "saved_frame_count": saved_count,
            "refresh_reason": refresh_reason,
            "backend": backend_used,
            "error": ffmpeg_error,
        }

    return {
        "status": "ok",
        "path": str(video_path),
        "source_frame_count": total_frames,
        "saved_frame_count": saved_count,
        "refresh_reason": refresh_reason,
        "backend": backend_used,
    }


def build_video_tasks(
    root,
    save_root,
    dataset_names=None,
    frame_stride=1,
    max_frames=None,
    overwrite=False,
    limit=None,
    hwaccel="qsv",
    decoder="auto",
    allow_cpu_fallback=False,
    jpeg_quality=2,
):
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
                hwaccel,
                decoder,
                allow_cpu_fallback,
                jpeg_quality,
            )
        )

    tasks.sort(key=lambda item: item[1])
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg was not found on PATH.")
    return ffmpeg_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos using FFmpeg hardware acceleration."
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
    parser.add_argument("--workers", type=int, default=4, help="Process count.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if manifest matches.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of videos to process.")
    parser.add_argument(
        "--hwaccel",
        default="qsv",
        help="FFmpeg hwaccel value. Defaults to qsv for Intel Quick Sync.",
    )
    parser.add_argument(
        "--decoder",
        default="auto",
        help="Optional explicit FFmpeg decoder, e.g. h264_qsv, hevc_qsv, vp9_qsv, or auto.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If hardware decode fails for a video, retry that video with CPU FFmpeg decode.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help="FFmpeg MJPEG quality. Lower is better quality. Typical range is 2-5.",
    )
    args = parser.parse_args()

    ffmpeg_path = check_ffmpeg()
    save_root = Path(args.save_root)

    print(
        f"Starting hardware frame extraction | datasets={args.datasets or 'all'} | frame_stride={args.frame_stride} | "
        f"max_frames={args.max_frames or 'all'} | workers={args.workers}"
    )
    print(
        f"FFmpeg backend | path={ffmpeg_path} | hwaccel={args.hwaccel} | "
        f"decoder={args.decoder} | cpu_fallback={args.allow_cpu_fallback}"
    )

    video_tasks = build_video_tasks(
        root=args.root,
        save_root=save_root,
        dataset_names=args.datasets,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        limit=args.limit,
        hwaccel=args.hwaccel,
        decoder=args.decoder,
        allow_cpu_fallback=args.allow_cpu_fallback,
        jpeg_quality=args.jpeg_quality,
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
            if item.get("error"):
                print(f"  error: {item['error']}")


if __name__ == "__main__":
    main()
