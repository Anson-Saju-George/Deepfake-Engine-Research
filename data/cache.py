"""Face-crop + landmark cache: packed WebDataset-shard format, writer/reader contract.

FIXTURE_PLAN.md T1.9, cache format B (packed) -- supersedes the format-A design
that shipped earlier in this repo's history (folder-per-video + one
landmarks.json + loose per-frame .jpg files, reusing the existing dtype="frame"
convention). Superseded because that's exactly the many-tiny-files problem a
packed format exists to avoid: ~12k videos x dozens of sampled frames each is
several hundred thousand loose files on disk.

Format (the WebDataset *naming convention* -- a plain tar, no dependency on the
`webdataset` package to read or write it; that package is an optional later
convenience for training-time multi-worker shard streaming, not required by
this module):
  each sampled frame's face crop is ONE tar-entry pair sharing a basename "key":
    <key>.jpg   -- crop image, JPEG bytes
    <key>.json  -- CrossRecord: bbox, landmarks, det_score, video_id, frame_idx,
                   label, dataset, and the extraction params, ALL in one record
                   next to the crop it describes (no second file to join against)
  key = "<dataset>__<video_stem>__<frame_idx:06d>", unique across the whole
  corpus so shards can be freely split, merged, or rebalanced later.

Cache-validity check is the same policy as format A, just relocated: still
keyed on detector/crop_size/margin/frame_stride param match + source mtime/size
(NOT a full source-video content hash -- hashing 12k videos' bytes is
expensive and this repo's videos aren't expected to mutate in place). Format A
read that from a landmarks.json sitting next to the crops; format B can't
cheaply scan every shard just to check one video, so a separate per-video
index (cache_index.json) is maintained alongside the shards instead.

Decode-failure policy for THIS pipeline (extraction, not training): a video
that fails to open/decode during extraction is skipped and logged by the
caller (proc/pre/extract_faces.py), never silently zero-filled -- the packed
cache only ever contains videos that extracted successfully. This is a
deliberately stricter, independent policy from DeepFakeDataset's existing
decode_failure_policy (default legacy_zero) for the raw-video training path;
the two do not affect each other.
"""
from __future__ import annotations

import json
import tarfile
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

CACHE_FORMAT_VERSION = 2  # v1 = format A (folder + landmarks.json), retired
CACHE_INDEX_NAME = "cache_index.json"
DEFAULT_SAMPLES_PER_SHARD = 2000


@dataclass
class FaceDetection:
    frame_idx: int
    bbox: list        # [x1, y1, x2, y2] in the CROP's own coordinate space, not the source frame
    landmarks: list    # flat list of [x, y] pairs, detector-native point count/order
    det_score: float


@dataclass
class CrossRecord:
    """One packed sample: crop + landmarks + det_score + ids + label, all in
    one record, matching the crop it's paired with in the tar. This is the
    unit T3's extractor writes and T2's dataloader eventually reads."""

    key: str            # "<dataset>__<video_stem>__<frame_idx:06d>", tar entry basename
    dataset: str
    video_id: str        # source video stem (not full path -- matches identity/split usage)
    label: int             # copied straight from DatasetBuilder's scan (real=1/fake=0, this
                             # repo's active convention -- this module carries it, does not
                             # interpret or convert it)
    frame_idx: int
    detection: FaceDetection
    detector: str
    crop_size: int
    margin: float
    frame_stride: int

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CrossRecord":
        det = d["detection"]
        detection = det if isinstance(det, FaceDetection) else FaceDetection(**det)
        return cls(
            key=d["key"],
            dataset=d["dataset"],
            video_id=d["video_id"],
            label=d["label"],
            frame_idx=d["frame_idx"],
            detection=detection,
            detector=d["detector"],
            crop_size=d["crop_size"],
            margin=d["margin"],
            frame_stride=d["frame_stride"],
        )


@dataclass
class VideoCacheIndexEntry:
    format_version: int
    source_path: str
    source_mtime: float
    source_size: int
    detector: str
    crop_size: int
    margin: float
    frame_stride: int
    created_at: float
    shard_ids: list   # tar shard filenames containing this video's samples
    keys: list          # this video's sample keys, in frame order

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoCacheIndexEntry":
        return cls(
            format_version=d.get("format_version", CACHE_FORMAT_VERSION),
            source_path=d["source_path"],
            source_mtime=d["source_mtime"],
            source_size=d["source_size"],
            detector=d["detector"],
            crop_size=d["crop_size"],
            margin=d["margin"],
            frame_stride=d["frame_stride"],
            created_at=d.get("created_at", 0.0),
            shard_ids=list(d.get("shard_ids", [])),
            keys=list(d.get("keys", [])),
        )


def video_cache_key(dataset: str, video_path: str | Path) -> str:
    return f"{dataset}__{Path(video_path).stem}"


# ---------------------------------------------------------------- index (validity checks)

def load_index(cache_root: str | Path) -> dict[str, VideoCacheIndexEntry]:
    path = Path(cache_root) / CACHE_INDEX_NAME
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: VideoCacheIndexEntry.from_dict(v) for k, v in raw.items()}


def save_index(cache_root: str | Path, index: dict[str, VideoCacheIndexEntry]) -> Path:
    path = Path(cache_root) / CACHE_INDEX_NAME
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: v.as_dict() for k, v in index.items()}, indent=2),
        encoding="utf-8",
    )
    return path


def is_cache_valid(
    index: dict[str, VideoCacheIndexEntry],
    dataset: str,
    source_path: str | Path,
    detector: str,
    crop_size: int,
    margin: float,
    frame_stride: int,
) -> bool:
    """A cache entry is valid only if it's indexed, every extraction param
    matches what's being requested now, AND the source file's mtime/size
    haven't changed since it was cached. Does not verify shard contents on
    disk (that would defeat the point of an index) -- a shard deleted out
    from under an intact index is a corruption case handled by the reader
    raising, not by this check."""
    entry = index.get(video_cache_key(dataset, source_path))
    if entry is None:
        return False
    if entry.format_version != CACHE_FORMAT_VERSION:
        return False
    if entry.detector != detector or entry.crop_size != crop_size:
        return False
    if entry.margin != margin or entry.frame_stride != frame_stride:
        return False

    source = Path(source_path)
    if not source.exists():
        return False
    stat = source.stat()
    if entry.source_mtime != stat.st_mtime or entry.source_size != stat.st_size:
        return False
    return True


# ---------------------------------------------------------------- writer

class ShardWriter:
    """Appends packed samples to size-bounded tar shards using the WebDataset
    naming convention (paired <key>.jpg / <key>.json tar entries). Pure
    stdlib tarfile -- no third-party dependency to write this format."""

    def __init__(self, cache_root: str | Path, samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self._shard_idx = self._next_shard_index()
        self._count_in_shard = 0
        self._tar: tarfile.TarFile | None = None
        self._open_new_shard()

    def _next_shard_index(self) -> int:
        existing = sorted(self.cache_root.glob("shard-*.tar"))
        if not existing:
            return 0
        last = existing[-1].stem.split("-")[-1]
        return int(last) + 1

    def _shard_path(self, idx: int) -> Path:
        return self.cache_root / f"shard-{idx:06d}.tar"

    def _open_new_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()
        self._tar = tarfile.open(self._shard_path(self._shard_idx), mode="w")
        self._count_in_shard = 0

    def current_shard_id(self) -> str:
        return self._shard_path(self._shard_idx).name

    def write_sample(self, record: CrossRecord, crop_jpeg_bytes: bytes) -> str:
        if self._count_in_shard >= self.samples_per_shard:
            self._shard_idx += 1
            self._open_new_shard()

        jpg_info = tarfile.TarInfo(name=f"{record.key}.jpg")
        jpg_info.size = len(crop_jpeg_bytes)
        self._tar.addfile(jpg_info, BytesIO(crop_jpeg_bytes))

        meta_bytes = json.dumps(record.as_dict()).encode("utf-8")
        json_info = tarfile.TarInfo(name=f"{record.key}.json")
        json_info.size = len(meta_bytes)
        self._tar.addfile(json_info, BytesIO(meta_bytes))

        self._count_in_shard += 1
        return self.current_shard_id()

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_video_to_cache(
    shard_writer: ShardWriter,
    index: dict[str, VideoCacheIndexEntry],
    *,
    dataset: str,
    source_path: str | Path,
    detector: str,
    crop_size: int,
    margin: float,
    frame_stride: int,
    detections_and_crops: list[tuple[FaceDetection, bytes]],
    label: int,
) -> VideoCacheIndexEntry:
    """Write every sampled-frame crop for one video into `shard_writer`,
    update `index` in place, return the new entry. Caller persists `index`
    via save_index -- batched across many videos, not per-video, so a crash
    mid-run doesn't require rescanning every shard to rebuild it (matches
    proc/pre/extract_faces.py's continuation-safe-rerun convention)."""
    video_key = video_cache_key(dataset, source_path)
    video_stem = Path(source_path).stem
    shard_ids: list[str] = []
    keys: list[str] = []

    for detection, crop_bytes in detections_and_crops:
        key = f"{dataset}__{video_stem}__{detection.frame_idx:06d}"
        record = CrossRecord(
            key=key,
            dataset=dataset,
            video_id=video_stem,
            label=label,
            frame_idx=detection.frame_idx,
            detection=detection,
            detector=detector,
            crop_size=crop_size,
            margin=margin,
            frame_stride=frame_stride,
        )
        shard_id = shard_writer.write_sample(record, crop_bytes)
        if shard_id not in shard_ids:
            shard_ids.append(shard_id)
        keys.append(key)

    stat = Path(source_path).stat()
    entry = VideoCacheIndexEntry(
        format_version=CACHE_FORMAT_VERSION,
        source_path=str(source_path),
        source_mtime=stat.st_mtime,
        source_size=stat.st_size,
        detector=detector,
        crop_size=crop_size,
        margin=margin,
        frame_stride=frame_stride,
        created_at=time.time(),
        shard_ids=shard_ids,
        keys=keys,
    )
    index[video_key] = entry
    return entry


# ---------------------------------------------------------------- reader

def iter_video_samples(cache_root: str | Path, index_entry: VideoCacheIndexEntry):
    """Yield (CrossRecord, crop_jpeg_bytes) for every sample belonging to one
    video, reading only the shard(s) it lives in -- the per-video random-access
    read path, matching DeepFakeDataset's one-video-per-__getitem__ shape. A
    sequential multi-worker streaming reader over ALL shards (the `webdataset`
    library's actual use case) is a later T2 wiring concern, not part of this
    format contract."""
    wanted = set(index_entry.keys)
    for shard_id in index_entry.shard_ids:
        shard_path = Path(cache_root) / shard_id
        with tarfile.open(shard_path, mode="r") as tar:
            pending: dict[str, dict[str, bytes]] = {}
            for member in tar.getmembers():
                stem, _, ext = member.name.rpartition(".")
                if stem not in wanted:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                pending.setdefault(stem, {})[ext] = extracted.read()
            for stem in index_entry.keys:
                parts = pending.get(stem)
                if not parts or "jpg" not in parts or "json" not in parts:
                    continue  # partial/corrupt pair -- caller's cache-validity check should catch this earlier
                record = CrossRecord.from_dict(json.loads(parts["json"].decode("utf-8")))
                yield record, parts["jpg"]


# ------------------------------------------------------- direct per-key reader (training)

# `iter_video_samples` above re-lists an entire shard's tar index
# (tar.getmembers()) on every call -- fine for a one-shot verification pass
# over a handful of videos, much too slow as the per-__getitem__ read path
# once training is sampling individual seq_len windows repeatedly across
# epochs (design question raised and approved -- see chat). This section
# keeps a small in-process cache of each shard's member index (built once via
# a single getmembers() scan, reused after that) so a training DataLoader
# worker's repeated reads from the same shard don't re-pay that scan cost.

_SHARD_INDEX_CACHE_LIMIT = 64  # open shard handles kept per worker process. Was 8;
# raised because a full-corpus pass can touch far more than 8 distinct shards
# (celeb-df-v2 alone: 1236 shards) -- 8 was fine for a handful-of-videos smoke
# test but caused real thrashing at production scale. Still bounded (not
# "= total shard count") since each entry also holds an open file handle.

# The bigger cost, found live during the first full-cache read-throughput
# pass (see chat): tarfile.getmembers() itself -- not cache misses -- is what's
# slow on drvfs (~960ms/shard on F:, ~550ms/shard on D:, ~58ms/shard on native
# ext4, measured directly). Walking ~4000 tar headers is a many-small-reads
# pattern, the exact case drvfs/9p is worst at (REPORT.md). Fix: after the
# first-ever scan of a shard, persist its {name: (offset, size)} map to a
# small JSON sidecar next to the shard. Every read after that loads the
# sidecar (one small file read) instead of re-walking tar headers, and reads
# member bytes via a raw seek+read on a plain file handle -- tarfile is only
# used for the one-time initial scan, never for the hot read path.
_SIDECAR_SUFFIX = ".offsets.json"


class _ShardIndex:
    """One shard's member-name -> (byte_offset, size) map. Built from a cached
    sidecar file when one exists (fast: one small JSON read); falls back to a
    full tarfile.getmembers() scan (slow on drvfs, see above) only the first
    time a shard is ever read, and writes the sidecar afterward so every
    subsequent read -- this process or a future one -- skips the scan."""

    def __init__(self, shard_path: Path):
        self.shard_path = shard_path
        sidecar_path = shard_path.with_name(shard_path.name + _SIDECAR_SUFFIX)
        offsets = self._load_sidecar(sidecar_path)
        if offsets is None:
            offsets = self._scan_and_write_sidecar(shard_path, sidecar_path)
        self._offsets = offsets  # {name: (offset, size)}
        self._fh = open(shard_path, "rb")

    @staticmethod
    def _load_sidecar(sidecar_path: Path) -> dict | None:
        if not sidecar_path.exists():
            return None
        try:
            raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
            return {name: tuple(pair) for name, pair in raw.items()}
        except Exception:
            return None  # corrupt/partial sidecar -- fall back to a real scan, don't trust it

    @staticmethod
    def _scan_and_write_sidecar(shard_path: Path, sidecar_path: Path) -> dict:
        with tarfile.open(shard_path, mode="r") as tar:
            members = tar.getmembers()
            offsets = {m.name: (m.offset_data, m.size) for m in members}
        try:
            sidecar_path.write_text(
                json.dumps({name: list(pair) for name, pair in offsets.items()}),
                encoding="utf-8",
            )
        except OSError:
            pass  # sidecar write failing (e.g. read-only mount) shouldn't break reads
        return offsets

    def has(self, key: str) -> bool:
        return f"{key}.jpg" in self._offsets

    def _read_raw(self, name: str) -> bytes:
        offset, size = self._offsets[name]
        self._fh.seek(offset)
        return self._fh.read(size)

    def read(self, key: str) -> tuple["CrossRecord", bytes]:
        if f"{key}.jpg" not in self._offsets or f"{key}.json" not in self._offsets:
            raise KeyError(f"key {key!r} has a partial/corrupt entry in {self.shard_path}")
        jpg_bytes = self._read_raw(f"{key}.jpg")
        json_bytes = self._read_raw(f"{key}.json")
        record = CrossRecord.from_dict(json.loads(json_bytes.decode("utf-8")))
        return record, jpg_bytes

    def close(self) -> None:
        self._fh.close()


_shard_index_cache: "OrderedDict[str, _ShardIndex]" = OrderedDict()


def _get_shard_index(shard_path: Path) -> _ShardIndex:
    cache_key = str(shard_path)
    existing = _shard_index_cache.get(cache_key)
    if existing is not None:
        _shard_index_cache.move_to_end(cache_key)
        return existing
    idx = _ShardIndex(shard_path)
    _shard_index_cache[cache_key] = idx
    if len(_shard_index_cache) > _SHARD_INDEX_CACHE_LIMIT:
        _, evicted = _shard_index_cache.popitem(last=False)
        evicted.close()
    return idx


def read_keys(
    cache_root: str | Path,
    index_entry: VideoCacheIndexEntry,
    keys: list[str],
) -> list[tuple[CrossRecord, bytes]]:
    """Direct per-key read for a specific ordered subset of one video's
    cached keys (e.g. one training clip window) -- the efficient read path
    `data/cache_dataset.py` uses, as opposed to `iter_video_samples`'s
    whole-video/whole-shard scan. A video's samples normally live in one
    shard (DEFAULT_SAMPLES_PER_SHARD comfortably exceeds one native-fps
    video's frame count in the common case), but this does not assume that --
    it searches index_entry.shard_ids in order for each key."""
    cache_root = Path(cache_root)
    out = []
    for key in keys:
        for shard_id in index_entry.shard_ids:
            shard_idx = _get_shard_index(cache_root / shard_id)
            if shard_idx.has(key):
                out.append(shard_idx.read(key))
                break
        else:
            raise KeyError(
                f"key {key!r} not found in any of this video's indexed shards {index_entry.shard_ids}"
            )
    return out
