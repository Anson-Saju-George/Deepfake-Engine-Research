"""Empirically find the max --workers for proc/pre/extract_faces.py that keeps
system-wide CPU utilization at or under a target (default 70%), instead of
guessing from os.cpu_count(). Runs short bursts at increasing worker counts on
a small sample of real videos, measuring actual CPU% via psutil throughout
each burst, and stops increasing once the target is exceeded.

Read-only against datasets. Writes nothing to any production cache -- each
burst uses its own throwaway scratch cache dir, deleted after the run.

Usage: python -m proc.pre.calibrate_workers --dataset celeb-df-v2 --target-cpu 70
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
from pathlib import Path

import psutil

from proc.pre.extract_faces import (
    DEFAULT_CROP_SIZE, DEFAULT_DETECTOR, DEFAULT_MARGIN, DEFAULT_MIN_DET_SCORE,
    _worker_compute, build_video_list,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_SAMPLE_SIZE = 12
DEFAULT_WORKER_COUNTS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


class _CpuSampler:
    """Background thread sampling system-wide CPU% every 0.3s for the
    duration of a burst. psutil.cpu_percent(interval=None) gives the delta
    since the last call, so this is a real running average across the burst,
    not a single instantaneous snapshot."""

    def __init__(self, interval=0.3):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        psutil.cpu_percent(interval=None)  # prime it -- first call is meaningless
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=self.interval))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def avg(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def peak(self):
        return max(self.samples) if self.samples else 0.0


def run_burst(videos, n_workers, detector, scratch_root: Path):
    jobs = [{
        "video_path": v[0], "dataset_name": v[1], "label": v[2],
        "max_frames": None, "crop_size": DEFAULT_CROP_SIZE, "margin": DEFAULT_MARGIN,
        "min_det_score": DEFAULT_MIN_DET_SCORE, "ctx_id": 0, "detector": detector,
    } for v in videos]

    with _CpuSampler() as sampler:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_worker_compute, job) for job in jobs]
            for f in as_completed(futures):
                f.result()  # drain, discard -- this burst never writes to cache
        wall_clock = time.time() - t0

    return {
        "n_workers": n_workers,
        "wall_clock_s": wall_clock,
        "videos_per_sec": len(videos) / wall_clock if wall_clock > 0 else 0.0,
        "avg_cpu_pct": sampler.avg(),
        "peak_cpu_pct": sampler.peak(),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate --workers for extract_faces.py against a CPU utilization cap.")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--dataset", default="celeb-df-v2")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--detector", choices=["retinaface", "yunet"], default=DEFAULT_DETECTOR)
    parser.add_argument("--target-cpu", type=float, default=70.0,
                         help="Stop increasing workers once avg CPU%% exceeds this.")
    parser.add_argument("--worker-counts", type=int, nargs="*", default=None,
                         help="Override the worker counts tried (default: 1,2,4,6,8,...,20).")
    args = parser.parse_args()

    worker_counts = args.worker_counts or DEFAULT_WORKER_COUNTS

    videos = build_video_list(root=args.root, dataset_names=[args.dataset], limit=args.sample_size)
    print(f"Calibration sample: {len(videos)} videos from {args.dataset}, detector={args.detector}, target_cpu={args.target_cpu}%")
    print(f"{'workers':>7s} {'avg_cpu%':>9s} {'peak_cpu%':>10s} {'wall_s':>8s} {'videos/sec':>11s}")

    scratch_root = Path(tempfile.mkdtemp(prefix="calibrate_workers_"))
    results = []
    chosen = 1
    try:
        for n in worker_counts:
            r = run_burst(videos, n, args.detector, scratch_root)
            results.append(r)
            print(f"{r['n_workers']:7d} {r['avg_cpu_pct']:9.1f} {r['peak_cpu_pct']:10.1f} "
                  f"{r['wall_clock_s']:8.2f} {r['videos_per_sec']:11.3f}")
            if r["avg_cpu_pct"] <= args.target_cpu:
                chosen = n
            else:
                print(f"  -> exceeded {args.target_cpu}% avg CPU at workers={n}, stopping")
                break
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    print()
    print(f"RECOMMENDED --workers {chosen} (highest tested count staying <= {args.target_cpu}% avg CPU)")
    print(f"Use it as: python -m proc.pre.extract_faces --datasets <ds> --cache-root <path> --workers {chosen}")


if __name__ == "__main__":
    main()
