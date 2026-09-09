"""Face detection + alignment + crop + landmark-sidecar extraction (FIXTURE_PLAN.md T3).

Two detectors are wired, selected via `--detector {retinaface,yunet}`:

- `retinaface` (via `insightface`, `buffalo_l`) -- the original choice (GPU-
  native, dense-enough landmarks for docs/code/sbi.py's convex-hull blend
  mask, strong detection on FF++ c23/Celeb-DF compression). Needs
  `insightface` + `onnxruntime-gpu` installed.
- `yunet` (via `cv2.FaceDetectorYN`, OpenCV's built-in detector) -- added
  during the smoke-test session that discovered `insightface`/`onnxruntime`
  are NOT installed in `/opt/ml` (the real production venv), so RetinaFace
  could not run there without a package install, which this task's
  constraints forbid. YuNet needs nothing beyond `opencv-python`, already a
  hard dependency. Model file: `test/_cache/models/face_detection_yunet_2023mar.onnx`
  (already present from `test/facedet/track_d_facedet.py`'s earlier
  detector-speed benchmark, no download needed here).

`test/results/REPORT.md`'s Track D benchmarked YuNet vs MTCNN, not vs
RetinaFace -- that alone never settled this file's detector choice.
**This smoke test settles it on evidence**: RetinaFace isn't runnable in
`/opt/ml` without an install, so YuNet is what actually gets evaluated here
(crop framing + SBI mask quality, eyeballed) -- if that holds up, YuNet wins
by default; if it doesn't, install RetinaFace's deps is the fallback, not the
starting point. Records `det_score` per detected face so low-confidence
detections can be filtered downstream, for both detectors.

**LOCKED RECIPE (do not alter without explicit re-approval -- see chat)**:
cache at NATIVE fps. Every decodable source frame gets a detection attempt;
no frame-skipping, no fps resampling, anywhere in this file. There used to be
a `--frame-stride` CLI knob defaulting to 4 (skip to every 4th frame) --
REMOVED. `NATIVE_FRAME_STRIDE = 1` below is the only value this file will
ever write into the cache; it exists as a named constant (not a free
parameter) purely so data/cache.py's cache-validity record has something
explicit to compare against, not because it's tunable again.

Mirrors proc/pre/pre_process_videos.py's conventions: discovers videos
through the same DatasetBuilder used everywhere else in this repo (not a
separate scan path), continuation-safe reruns. Cache format/validity is
data/cache.py's -- packed WebDataset-shard format (format B), superseding
this file's original per-video-folder + landmarks.json write path. See that
module's docstring for the format and why it changed.

insightface + onnxruntime are NOT installed in the environment this was
written in (no GPU/torch/cv2 sandbox available at authoring time -- see
FIXTURE_PLAN.md's Quality Gates note on this). The detection call itself
(FaceAnalysis.get()) has not been executed end-to-end; only the surrounding
discovery/cache/crop-geometry logic was verified against the real corpus and
via synthetic tests (cache round-trip: see chat, not checked in as a
persisted test file). Smoke-test this against a small --limit before a full
run:

  python -m proc.pre.extract_faces --datasets celeb-df-v2 --limit 5
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

try:
    from data.dataloader import DatasetBuilder
except ImportError:
    from dataloader import DatasetBuilder

from data.cache import (
    FaceDetection,
    ShardWriter,
    is_cache_valid,
    load_index,
    save_index,
    write_video_to_cache,
)

DEFAULT_ROOT = "datasets"
DEFAULT_CACHE_ROOT = Path("datasets/preprocessed_frames")
DEFAULT_DETECTOR = "yunet"
DETECTOR_NAMES = {
    "retinaface": "retinaface-insightface",
    "yunet": "yunet-cv2",
}
YUNET_MODEL_PATH = Path(__file__).resolve().parents[2] / "test" / "_cache" / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_CROP_SIZE = 256
DEFAULT_MARGIN = 1.3
DEFAULT_MIN_DET_SCORE = 0.5
JPEG_QUALITY = 90  # was 95; measured ~24.5% smaller at 90 with no expected visible
                    # quality loss for a 256x256 face crop (see chat's storage-projection pass)
# Locked recipe: cache stores one detection attempt per ORIGINAL frame, always.
# Not a CLI flag on purpose -- see module docstring.
NATIVE_FRAME_STRIDE = 1

_face_app = None  # lazily constructed, one per process (ProcessPoolExecutor-safe)
_face_app_kind = None  # which detector _face_app was built for -- catches a stale
                        # cached app if a future caller switches detectors mid-process


class _YuNetFace:
    """Matches the subset of insightface's Face object interface that
    extract_faces_for_video actually reads (.bbox, .kps, .det_score) -- so
    the crop-geometry/landmark-translation code downstream doesn't need to
    know which detector produced a detection."""
    __slots__ = ("bbox", "kps", "det_score")

    def __init__(self, bbox, kps, det_score):
        self.bbox = bbox
        self.kps = kps
        self.det_score = det_score


DETECT_MAX_SIDE = 1600  # was 640 (insightface's det_size convention) -- raised after
                        # a targeted tuning experiment (test/_cache/tune_resize_cap.py,
                        # see chat) found 640 caused a real recall regression on 1080p
                        # FF++ content (hugging_happy 0.998->0.962, exit_phone_room
                        # 0.990->0.826), not just the intended speed win. At 1600, both
                        # recover to ~baseline (0.983-1.000). celeb-df-v2 is unaffected
                        # at any cap tested (native res 944px < any cap here, so it's
                        # never resized) -- confirmed safe for the run this constant
                        # shipped with.
                        # NOT FIXED by any cap tested, including 1600: real-ai-videos
                        # (real(1).mp4 stuck at 0.0-0.067, ai(1).mp4 flat ~0.38-0.53
                        # regardless of cap) -- this is content-driven (small/angled/
                        # off-frame faces), not a resolution-scale artifact. Resize-cap
                        # tuning cannot fix it; needs a different remedy if it matters
                        # (different detector, lower min_det_score, or accept the gaps
                        # on this 66-video/0.5%-of-corpus dataset) before real-ai-videos
                        # extraction, not before FF++'s.


class _YuNetFaceApp:
    """Drop-in replacement for insightface's FaceAnalysis, backed by
    cv2.FaceDetectorYN -- no dependency beyond opencv-python (already
    required). `.get(frame_bgr)` returns a list of `_YuNetFace`, matching
    FaceAnalysis.get()'s return shape closely enough for this file's use.

    Detects on a resized copy (long side capped at DETECT_MAX_SIDE), then
    scales bbox/landmarks back to the ORIGINAL frame's coordinate space
    before returning -- downstream code (square_crop_box,
    translate_landmarks_to_crop) never needs to know detection happened at a
    different resolution than the source frame."""

    def __init__(self, model_path, max_side=DETECT_MAX_SIDE):
        self._model_path = str(model_path)
        self._max_side = max_side
        self._detector = None
        self._detect_size = None

    def _ensure(self, w, h):
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(self._model_path, "", (w, h))
            self._detect_size = (w, h)
        elif self._detect_size != (w, h):
            self._detector.setInputSize((w, h))
            self._detect_size = (w, h)

    def get(self, frame_bgr):
        orig_h, orig_w = frame_bgr.shape[:2]
        scale = min(1.0, self._max_side / max(orig_w, orig_h))
        if scale < 1.0:
            det_w, det_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
            detect_frame = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_w, det_h = orig_w, orig_h
            detect_frame = frame_bgr

        self._ensure(det_w, det_h)
        _retval, faces = self._detector.detect(detect_frame)
        if faces is None:
            return []
        inv_scale = 1.0 / scale if scale < 1.0 else 1.0
        out = []
        for row in faces:
            # cv2.FaceDetectorYN row layout: x,y,w,h, then 5 landmark (x,y)
            # pairs (right eye, left eye, nose tip, right mouth corner, left
            # mouth corner), then score -- 15 values total. All spatial
            # values are in the RESIZED frame's coordinates -- scale back to
            # native before constructing the Face object.
            row = row.copy()
            row[0:14] *= inv_scale
            x, y, fw, fh = row[0:4]
            bbox = [float(x), float(y), float(x + fw), float(y + fh)]
            kps = np.asarray(row[4:14], dtype=float).reshape(5, 2)
            score = float(row[14])
            out.append(_YuNetFace(bbox=bbox, kps=kps, det_score=score))
        return out


def _get_face_app(detector: str = DEFAULT_DETECTOR, ctx_id: int = 0, det_size: int = 640):
    """Lazy import + lazy construction: importing this module must not
    require insightface to be installed (matches the repo's existing
    lazy-import pattern for optional heavy deps, e.g. profile_model_complexity.py's
    thop/fvcore). Only actually running extraction with detector="retinaface"
    requires it; "yunet" only needs cv2, already imported at module load."""
    global _face_app, _face_app_kind
    if _face_app is not None and _face_app_kind == detector:
        return _face_app

    if detector == "yunet":
        if not YUNET_MODEL_PATH.exists():
            raise RuntimeError(
                f"YuNet model file not found at {YUNET_MODEL_PATH}. It should already "
                f"exist from test/facedet/track_d_facedet.py's earlier benchmark run -- "
                f"if it's genuinely missing, that benchmark's model download step needs "
                f"to be re-run, not silently substituted here."
            )
        app = _YuNetFaceApp(YUNET_MODEL_PATH)
    elif detector == "retinaface":
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface is not installed. Install it (and onnxruntime-gpu for "
                "GPU inference) before running with --detector retinaface: "
                "pip install insightface onnxruntime-gpu"
            ) from exc
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    else:
        raise ValueError(f"Unknown detector: {detector!r} (expected 'retinaface' or 'yunet')")

    _face_app = app
    _face_app_kind = detector
    return app


def compute_frame_indices(total_frames: int, max_frames: int | None = None) -> list[int]:
    """Locked recipe: every original frame, always (NATIVE_FRAME_STRIDE=1) --
    no skip-pattern parameter exists here anymore. `max_frames` is an
    explicit, opt-in safety cap (default None = uncapped, full native
    coverage); passing it is a deliberate departure from strict native-frame
    coverage for a specific run, not a normal part of this recipe."""
    if total_frames <= 0:
        return []
    indices = list(range(total_frames))
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        sampled = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
        indices = [indices[i] for i in sampled]
    return indices


def square_crop_box(bbox, frame_width: int, frame_height: int, margin: float) -> tuple[int, int, int, int]:
    """Expand a detector bbox into a square crop box with the given margin,
    centered on the bbox, clamped to the frame boundary. Returns (x1, y1, x2, y2)
    in the SOURCE frame's coordinate space (caller translates landmarks into
    the crop's local space after this)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * margin
    half = side / 2.0

    left = cx - half
    top = cy - half
    right = cx + half
    bottom = cy + half

    # clamp without distorting the square aspect ratio: shift the box fully
    # inside the frame rather than shrinking it asymmetrically
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > frame_width:
        left -= (right - frame_width)
        right = frame_width
    if bottom > frame_height:
        top -= (bottom - frame_height)
        bottom = frame_height

    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(frame_width, int(round(right)))
    bottom = min(frame_height, int(round(bottom)))
    return left, top, right, bottom


def translate_landmarks_to_crop(landmarks, box, crop_size: int) -> list:
    """Map detector-native landmark points (in source-frame coordinates) into
    the resized crop's coordinate space."""
    left, top, right, bottom = box
    box_w = max(right - left, 1)
    box_h = max(bottom - top, 1)
    scale_x = crop_size / box_w
    scale_y = crop_size / box_h
    return [[(x - left) * scale_x, (y - top) * scale_y] for x, y in landmarks]


def _compute_detections_for_video(
    video_path: str,
    max_frames: int | None,
    crop_size: int,
    margin: float,
    min_det_score: float,
    ctx_id: int,
    detector: str,
) -> tuple[str, int, list[tuple[FaceDetection, bytes]]]:
    """Pure compute: decode + detect + crop + encode for ONE video. No shared
    state (no ShardWriter, no index) -- safe to run inside a worker process.
    Returns (status, frame_count, detections_and_crops). Split out of
    extract_faces_for_video specifically so multiprocessing has a clean unit
    of work: workers compute, the main process is the sole cache writer
    (tarfile isn't safe for concurrent writers, and a single writer also
    keeps shard/index bookkeeping simple -- see main()'s pool loop)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "failed_open", 0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = compute_frame_indices(total_frames, max_frames=max_frames)
    if not indices:
        cap.release()
        return "failed_metadata", total_frames, []

    app = _get_face_app(detector=detector, ctx_id=ctx_id)

    detections_and_crops: list[tuple[FaceDetection, bytes]] = []
    # `indices` is sorted ascending by construction (compute_frame_indices).
    # Under the locked recipe (max_frames=None) it's ALWAYS range(total_frames)
    # -- strictly consecutive -- so seeking before every read (the original
    # code, inherited from the --frame-stride>1 era where jumps were real) is
    # pure overhead: cap.set() on a compressed codec forces a seek-to-keyframe
    # + decode-forward instead of a cheap sequential read, and doing that on
    # EVERY frame instead of only every 4th made extraction dramatically
    # slower after --frame-stride was removed (caught live during the smoke
    # test -- see chat). Only seek when the next wanted frame isn't the one
    # a plain sequential read would already land on.
    next_sequential_idx = 0
    for frame_idx in indices:
        if frame_idx != next_sequential_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        next_sequential_idx = frame_idx + 1
        if not ret or frame is None:
            continue

        faces = app.get(frame)  # BGR numpy array in, list of Face objects out
        if not faces:
            continue

        # keep only the highest-confidence face per frame -- matches the
        # single-primary-subject assumption used elsewhere in this pipeline
        # (identity-per-video, not per-face); revisit if a future dataset
        # needs multi-face-per-frame extraction.
        face = max(faces, key=lambda f: float(f.det_score))
        if float(face.det_score) < min_det_score:
            continue

        frame_h, frame_w = frame.shape[:2]
        box = square_crop_box(face.bbox, frame_w, frame_h, margin)
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            continue

        crop = frame[top:bottom, left:right]
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        crop_bytes = encoded.tobytes()

        landmarks_in_crop = translate_landmarks_to_crop(face.kps.tolist(), box, crop_size)
        detections_and_crops.append((
            FaceDetection(
                frame_idx=int(frame_idx),
                bbox=[0, 0, crop_size, crop_size],
                landmarks=landmarks_in_crop,
                det_score=float(face.det_score),
            ),
            crop_bytes,
        ))

    cap.release()

    if not detections_and_crops:
        return "failed_no_faces", total_frames, []
    return "ok", total_frames, detections_and_crops


def _worker_compute(job: dict) -> dict:
    """Module-level (picklable) entry point for ProcessPoolExecutor workers.
    Pure compute, no shared state -- each worker process lazily builds its
    own detector instance on first call via _get_face_app's existing
    per-process global cache (safe: separate process, separate memory,
    no cross-process sharing of the cv2/insightface detector object, which
    wouldn't pickle anyway). Returns everything the main process needs to
    write the result to cache, or skip it."""
    status, total_frames, detections_and_crops = _compute_detections_for_video(
        job["video_path"], job["max_frames"], job["crop_size"], job["margin"],
        job["min_det_score"], job["ctx_id"], job["detector"],
    )
    return {
        "video_path": job["video_path"],
        "dataset_name": job["dataset_name"],
        "label": job["label"],
        "status": status,
        "frame_count": total_frames,
        "detections_and_crops": detections_and_crops,
    }


def extract_faces_for_video(
    video_path: str,
    dataset_name: str,
    label: int,
    shard_writer: ShardWriter,
    index: dict,
    max_frames: int | None,
    crop_size: int,
    margin: float,
    min_det_score: float,
    ctx_id: int,
    overwrite: bool,
    detector: str = DEFAULT_DETECTOR,
) -> dict:
    """Single-process path: compute + write in one call. Kept for backward
    compatibility (the smoke-test driver and any single-video caller use
    this); main()'s --workers>1 path calls _compute_detections_for_video and
    write_video_to_cache separately instead -- see main()."""
    detector_name = DETECTOR_NAMES[detector]
    if not overwrite and is_cache_valid(index, dataset_name, video_path, detector_name, crop_size, margin, NATIVE_FRAME_STRIDE):
        return {"status": "skipped", "path": video_path}

    status, total_frames, detections_and_crops = _compute_detections_for_video(
        video_path, max_frames, crop_size, margin, min_det_score, ctx_id, detector,
    )
    if status != "ok":
        return {"status": status, "path": video_path, "frame_count": total_frames}

    write_video_to_cache(
        shard_writer,
        index,
        dataset=dataset_name,
        source_path=video_path,
        detector=detector_name,
        crop_size=crop_size,
        margin=margin,
        frame_stride=NATIVE_FRAME_STRIDE,
        detections_and_crops=detections_and_crops,
        label=label,
    )

    return {"status": "ok", "path": video_path, "faces_extracted": len(detections_and_crops)}


def build_video_list(root=None, dataset_names=None, limit=None, only_relpaths=None) -> list[tuple[str, str, int]]:
    """`only_relpaths`: an explicit stratified-sample selection (e.g. the
    smoke-test list), matched by whether a record's path ENDS WITH the given
    relative path (records use absolute paths; this avoids needing to know
    the exact `datasets/videos/` root prefix on whatever machine this runs
    on). Raises if any requested relpath isn't found -- fail loud rather than
    silently extracting fewer videos than approved."""
    builder = DatasetBuilder(root=root)
    builder.build()
    allowed = set(dataset_names) if dataset_names else None

    videos = []
    for record in builder.records:
        if record["dtype"] != "video":
            continue
        if allowed and record["dataset"] not in allowed:
            continue
        videos.append((record["path"], record["dataset"], record["label"]))

    if only_relpaths is not None:
        normalized = [str(Path(p)).replace("\\", "/") for p in only_relpaths]
        matched = []
        for relpath in normalized:
            hits = [v for v in videos if v[0].replace("\\", "/").endswith(relpath)]
            if not hits:
                raise ValueError(f"No video found on disk matching requested path: {relpath!r}")
            if len(hits) > 1:
                raise ValueError(f"Ambiguous match for {relpath!r}: {[h[0] for h in hits]}")
            matched.append(hits[0])
        return matched

    videos.sort()
    if limit is not None:
        videos = videos[:limit]
    return videos


def main():
    parser = argparse.ArgumentParser(description="Extract face crops + landmark sidecars via RetinaFace/insightface.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--video-paths", nargs="*", default=None,
                         help="Explicit relative-path selection (e.g. the smoke-test "
                              "stratified sample). Matched by path suffix. Overrides "
                              "--datasets/--limit when given.")
    parser.add_argument("--detector", choices=["retinaface", "yunet"], default=DEFAULT_DETECTOR,
                         help="retinaface needs insightface+onnxruntime-gpu installed; "
                              "yunet needs only cv2 (already a hard dependency).")
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Optional safety cap on frames processed per video. Default (None) is "
             "uncapped native coverage -- the locked recipe. Setting this is a "
             "deliberate departure from strict native-fps caching for one run.",
    )
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--min-det-score", type=float, default=DEFAULT_MIN_DET_SCORE)
    parser.add_argument("--ctx-id", type=int, default=0, help="insightface ctx_id: >=0 GPU device index, -1 CPU.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--samples-per-shard", type=int, default=2000,
        help="Packed-cache tar shard size, in samples (crop+metadata pairs).",
    )
    parser.add_argument(
        "--index-flush-every", type=int, default=50,
        help="Persist cache_index.json every N videos, so a crash mid-run doesn't "
             "lose more than this many videos' worth of index entries (shards "
             "themselves are never rewritten, only appended to).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes for detection (CPU-bound -- decode+YuNet "
             "per video). 1 = original sequential path, unchanged. >1 uses "
             "ProcessPoolExecutor: workers compute only, the main process is the "
             "sole cache writer (tarfile isn't safe for concurrent writers). "
             "Pick this empirically with proc/pre/calibrate_workers.py, which caps "
             "at a target CPU utilization rather than assuming cpu_count() is safe.",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    videos = build_video_list(root=args.root, dataset_names=args.datasets, limit=args.limit,
                               only_relpaths=args.video_paths)
    print(f"Videos discovered: {len(videos)} | cache_root={cache_root} | detector={args.detector} | workers={args.workers}")

    index = load_index(cache_root)
    results = []

    if args.workers <= 1:
        with ShardWriter(cache_root, samples_per_shard=args.samples_per_shard) as shard_writer:
            for i, (video_path, dataset_name, label) in enumerate(videos):
                result = extract_faces_for_video(
                    video_path=video_path,
                    dataset_name=dataset_name,
                    label=label,
                    shard_writer=shard_writer,
                    index=index,
                    max_frames=args.max_frames,
                    crop_size=args.crop_size,
                    detector=args.detector,
                    margin=args.margin,
                    min_det_score=args.min_det_score,
                    ctx_id=args.ctx_id,
                    overwrite=args.overwrite,
                )
                results.append(result)
                print(f"{result['status']:16} | {result['path']}")

                if result["status"] == "ok" and (i + 1) % args.index_flush_every == 0:
                    save_index(cache_root, index)
        save_index(cache_root, index)
    else:
        # Parallel path: filter cache-valid videos in the main process first
        # (no point paying worker dispatch cost for a skip), submit the rest
        # to a process pool, then write results to cache SEQUENTIALLY in the
        # main process as they complete -- ShardWriter/index are single-writer
        # by design, matching data/cache.py's contract.
        detector_name = DETECTOR_NAMES[args.detector]
        pending = []
        for video_path, dataset_name, label in videos:
            if not args.overwrite and is_cache_valid(
                index, dataset_name, video_path, detector_name, args.crop_size, args.margin, NATIVE_FRAME_STRIDE
            ):
                results.append({"status": "skipped", "path": video_path})
                continue
            pending.append({
                "video_path": video_path, "dataset_name": dataset_name, "label": label,
                "max_frames": args.max_frames, "crop_size": args.crop_size, "margin": args.margin,
                "min_det_score": args.min_det_score, "ctx_id": args.ctx_id, "detector": args.detector,
            })
        print(f"{len(results)} already cache-valid (skipped), {len(pending)} to extract with {args.workers} workers")

        with ShardWriter(cache_root, samples_per_shard=args.samples_per_shard) as shard_writer:
            n_written_since_flush = 0
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_worker_compute, job): job for job in pending}
                for future in as_completed(futures):
                    r = future.result()
                    if r["status"] == "ok":
                        write_video_to_cache(
                            shard_writer, index,
                            dataset=r["dataset_name"], source_path=r["video_path"],
                            detector=detector_name, crop_size=args.crop_size, margin=args.margin,
                            frame_stride=NATIVE_FRAME_STRIDE,
                            detections_and_crops=r["detections_and_crops"], label=r["label"],
                        )
                        n_written_since_flush += 1
                        if n_written_since_flush >= args.index_flush_every:
                            save_index(cache_root, index)
                            n_written_since_flush = 0
                    results.append({"status": r["status"], "path": r["video_path"]})
                    print(f"{r['status']:16} | {r['video_path']}")
        save_index(cache_root, index)

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] not in {"ok", "skipped"}]
    print(f"\nFinished. Extracted: {ok} | Skipped: {skipped} | Failed: {len(failed)}")
    if failed:
        print("Sample failures:")
        for item in failed[:20]:
            print(f"{item['status']} | {item['path']}")


if __name__ == "__main__":
    main()
