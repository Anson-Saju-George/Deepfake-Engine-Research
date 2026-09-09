"""PyTorch Dataset reading directly from data/cache.py's packed native-fps
cache. Never decodes raw video -- every frame here was already extracted and
cropped by proc/pre/extract_faces.py. Exists to answer one question: is the
cache-read training loop actually decode-free and fast (see chat's P4-P6
verification + throughput benchmark).

THE LOCKED RECIPE (do not alter without explicit re-approval -- see chat):
  - native fps, no resampling anywhere (guaranteed upstream by
    proc/pre/extract_faces.py's NATIVE_FRAME_STRIDE=1 lock)
  - stride = 1, always, on the CACHED frame sequence. Note precisely what
    that means here: a video's cached keys (VideoCacheIndexEntry.keys) are in
    extraction frame order, but extraction skips a frame when no face was
    detected on it (extract_faces_for_video: "if not faces: continue") -- so
    "stride=1 on cached frames" means consecutive entries in the CACHED
    sequence, which can correspond to non-consecutive original frame_idx
    values when a no-face frame was skipped. That is expected and is NOT the
    same thing as a stride>1 sampling choice -- it's a property of what got
    detected, not a sampling decision. Track D found ~1.00-1.05 faces/frame
    on this corpus, so gaps should be rare in practice; the verification
    script should report actual frame_idx gaps it finds, not assume zero.
  - seq_len in {8, 16} -- the only values under test, no others
  - accepted limitation: real-time clip span = seq_len / src_fps, VARIES by
    source video. Do not "fix" this with resampling.
  - short-clip policy: loop-pad (repeat FROM THE START, not from the last
    frame -- deliberately different from data/dataloader.py's raw-video
    short-clip padding, which repeats the last frame; do not conflate the
    two). Never drop a video for being short. Counted and logged per dataset
    via PadLog below.
  - eval: non-overlapping seq_len windows tiling the full cached sequence,
    one score per window, mean-aggregated to one video-level prediction by
    the caller (this Dataset yields one window per sample; aggregation-by-
    video_id happens at the eval-loop level, matching this repo's existing
    pattern in train/eval_predictions_common.py / build_multi_clip_eval_samples
    rather than inside a Dataset's __getitem__).
"""
from __future__ import annotations

import io
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from data.cache import VideoCacheIndexEntry, read_keys

DEFAULT_SEQ_LENS = (8, 16)


class PadLog:
    """Counts + logs every video that triggered loop-pad, grouped by dataset,
    per the locked recipe's "count + log padded videos per dataset"
    requirement. Not a metric -- a plain audit trail."""

    def __init__(self):
        self.counts: dict[str, int] = defaultdict(int)
        self._events: list[dict] = []

    def record(self, dataset: str, video_id: str, n_cached: int, seq_len: int) -> None:
        self.counts[dataset] += 1
        self._events.append({
            "dataset": dataset,
            "video_id": video_id,
            "n_cached_frames": n_cached,
            "seq_len": seq_len,
            "ts": time.time(),
        })

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event) + "\n")
        return path

    def summary(self) -> dict[str, int]:
        return dict(self.counts)


def _loop_pad_from_start(keys: list[str], seq_len: int) -> list[str]:
    """Repeat the key sequence from its start until seq_len is reached.
    n=3, seq_len=8 -> keys[0],keys[1],keys[2],keys[0],keys[1],keys[2],keys[0],keys[1]
    -- distinct on purpose from data/dataloader.py's repeat-last-frame
    short-clip fallback (see module docstring)."""
    if not keys:
        raise ValueError(
            "cannot loop-pad an empty key list -- extraction never writes an "
            "index entry for a video with zero cached frames"
        )
    return [keys[i % len(keys)] for i in range(seq_len)]


def _decode_jpeg_tensor(jpeg_bytes: bytes, transform=None) -> torch.Tensor:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    if transform is not None:
        return transform(img)
    return TF.to_tensor(img)


@dataclass
class CacheSample:
    """One resolved training/eval sample: which video, which exact seq_len
    keys (already stride-1-consecutive-in-cached-order, already loop-padded
    if needed), and its label. Carries shard_ids directly (not just a
    video_id to look up later) so this Dataset never needs a live reference
    to the full index dict -- keeps it self-contained and trivially
    picklable across DataLoader worker processes."""
    dataset: str
    video_id: str
    keys: list[str]
    shard_ids: list[str]
    label: int
    padded: bool


class NativeCacheClipDataset(Dataset):
    """Resolves a flat list of CacheSample (built by build_train_samples/
    build_eval_samples below) into (frames_tensor, label) pairs, reading
    crop bytes via data/cache.py's direct per-key reader -- no raw video
    decode anywhere in this class."""

    def __init__(self, cache_root: str | Path, samples: list[CacheSample], transform=None):
        self.cache_root = Path(cache_root)
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        # read_keys only needs shard_ids off the index entry; everything else
        # in VideoCacheIndexEntry is validity metadata this call doesn't use.
        entry_stub = VideoCacheIndexEntry(
            format_version=0, source_path="", source_mtime=0.0, source_size=0,
            detector="", crop_size=0, margin=0.0, frame_stride=1, created_at=0.0,
            shard_ids=sample.shard_ids, keys=sample.keys,
        )
        pairs = read_keys(self.cache_root, entry_stub, sample.keys)
        frames = torch.stack([_decode_jpeg_tensor(jpg, self.transform) for _, jpg in pairs])
        return frames, sample.label


def build_train_samples(
    index: dict[str, VideoCacheIndexEntry],
    dataset_labels: dict[str, tuple[str, int]],
    seq_len: int,
    seed: int | None = None,
    pad_log: PadLog | None = None,
) -> list[CacheSample]:
    """One random contiguous stride-1 window per video, loop-padded if short.
    `dataset_labels` maps video_key -> (dataset_name, label) since
    VideoCacheIndexEntry itself doesn't carry a label (CrossRecord does, per
    crop -- fetched lazily via read_keys, not duplicated onto the index)."""
    rng = random.Random(seed)
    samples = []
    for video_key, entry in index.items():
        dataset_name, label = dataset_labels[video_key]
        keys = entry.keys
        padded = len(keys) < seq_len
        if padded:
            window = _loop_pad_from_start(keys, seq_len)
            if pad_log is not None:
                pad_log.record(dataset_name, video_key, len(keys), seq_len)
        else:
            start = rng.randint(0, len(keys) - seq_len)
            window = keys[start:start + seq_len]
        samples.append(CacheSample(
            dataset=dataset_name, video_id=video_key, keys=window,
            shard_ids=entry.shard_ids, label=label, padded=padded,
        ))
    return samples


def build_eval_samples(
    index: dict[str, VideoCacheIndexEntry],
    dataset_labels: dict[str, tuple[str, int]],
    seq_len: int,
    pad_log: PadLog | None = None,
) -> tuple[list[CacheSample], list[str]]:
    """Non-overlapping seq_len windows tiling each video's full cached
    sequence -- reuses NO frame across windows within a video, per the locked
    recipe. Returns (samples, video_ids) where video_ids[i] identifies which
    source video samples[i] came from, matching this repo's existing
    multi-clip-eval return shape (DatasetBuilder.build_multi_clip_eval_samples)
    so the caller aggregates per-window predictions into one video-level
    score with the same mean-of-prob_real pattern already used elsewhere."""
    samples = []
    video_ids = []
    for video_key, entry in index.items():
        dataset_name, label = dataset_labels[video_key]
        keys = entry.keys
        if len(keys) < seq_len:
            window = _loop_pad_from_start(keys, seq_len)
            if pad_log is not None:
                pad_log.record(dataset_name, video_key, len(keys), seq_len)
            samples.append(CacheSample(
                dataset=dataset_name, video_id=video_key, keys=window,
                shard_ids=entry.shard_ids, label=label, padded=True,
            ))
            video_ids.append(video_key)
            continue

        n_windows = len(keys) // seq_len  # non-overlapping -> floor division, no reuse
        for w in range(n_windows):
            window = keys[w * seq_len:(w + 1) * seq_len]
            samples.append(CacheSample(
                dataset=dataset_name, video_id=video_key, keys=window,
                shard_ids=entry.shard_ids, label=label, padded=False,
            ))
            video_ids.append(video_key)
    return samples, video_ids
