"""Audit raw-video decode failures and write a reviewable suspect list."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torchvision.transforms as transforms

from data.dataloader import DatasetBuilder, DeepFakeDataset


DATASET_SCOPE_MAP = {
    "celebdf": ["celeb-df-v2"],
    "ffpp": ["faceforensics++"],
    "video_combined": ["celeb-df-v2", "faceforensics++"],
    "real_ai_videos": ["real-ai-videos"],
    "video_all": ["celeb-df-v2", "faceforensics++", "real-ai-videos"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit raw videos for decode failures.")
    parser.add_argument(
        "--dataset-scope",
        choices=sorted(DATASET_SCOPE_MAP.keys()),
        default="video_combined",
        help="Named video dataset scope to audit.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional explicit dataset names. Overrides --dataset-scope.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sequence", "both"],
        default="both",
        help="Which raw-video decode paths to audit.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Contiguous clip length used for sequence audit mode.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=["auto", "decord", "decord_cpu", "cv2", "ffmpeg", "ffmpeg_qsv", "ffmpeg_d3d11va"],
        default="ffmpeg_qsv",
        help="Decode backend to audit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of raw videos to audit.",
    )
    parser.add_argument(
        "--log-path",
        default="train/video/bad_video_log.jsonl",
        help="JSONL log path where decode failures are appended.",
    )
    return parser.parse_args()


def load_log_records(log_path: Path):
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def audit_dataset(dataset_names: list[str], mode: str, seq_len: int, limit: int | None) -> None:
    builder = DatasetBuilder(root="datasets", seed=42)
    builder.build()
    records = builder._filter_records(dtype="video", dataset_names=dataset_names)
    if limit is not None:
        records = records[:limit]

    samples = [(record["path"], record["label"], "video") for record in records]
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = DeepFakeDataset(samples, transform=tf, mode=mode, seq_len=seq_len, clip_sampling="center")

    print(f"Audit mode: {mode}")
    print(f"Datasets: {dataset_names}")
    print(f"Videos to audit: {len(samples)}")

    success = 0
    for idx in range(len(dataset)):
        sample, _label = dataset[idx]
        if mode == "single":
            ok = sample is not None and tuple(sample.shape) == (3, 224, 224) and float(sample.abs().sum()) > 0.0
        else:
            ok = sample is not None and sample.ndim == 4 and float(sample.abs().sum()) > 0.0
        success += int(ok)
        if (idx + 1) % 500 == 0 or idx + 1 == len(dataset):
            print(f"checked={idx + 1}/{len(dataset)} | success={success}")


def main():
    args = parse_args()
    dataset_names = args.datasets or DATASET_SCOPE_MAP[args.dataset_scope]
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    os.environ["DF_BAD_VIDEO_LOG_PATH"] = str(log_path)
    os.environ["DF_VIDEO_DECODE_BACKEND"] = args.decode_backend

    modes = ["single", "sequence"] if args.mode == "both" else [args.mode]
    for mode in modes:
        audit_dataset(dataset_names=dataset_names, mode=mode, seq_len=args.seq_len, limit=args.limit)

    records = load_log_records(log_path)
    unique_paths = sorted({record.get("path") for record in records if record.get("path")})
    stage_counts = Counter(record.get("stage", "unknown") for record in records)

    summary = {
        "log_path": str(log_path),
        "decode_backend": args.decode_backend,
        "dataset_names": dataset_names,
        "mode": args.mode,
        "seq_len": args.seq_len,
        "checked_limit": args.limit,
        "total_events": len(records),
        "unique_bad_videos": len(unique_paths),
        "stage_counts": dict(stage_counts),
    }

    summary_path = log_path.with_name("bad_video_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    suspect_path = log_path.with_name("bad_video_suspects.txt")
    suspect_path.write_text("\n".join(unique_paths) + ("\n" if unique_paths else ""), encoding="utf-8")

    print("\nBad video summary")
    print(json.dumps(summary, indent=2))
    print(f"Suspect list: {suspect_path}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
