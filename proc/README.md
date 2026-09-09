# proc/

Download, integrity, and frame-extraction utilities (FIXTURE_PLAN.md T3).

Layout is categorized because this directory will keep growing as more
datasets and preprocessing steps get added:

```
proc/
  manifest.py               # cross-cutting: integrity manifest schema
  integrity_scan.py          # cross-cutting: builds a manifest from data/dataset_run.py's validator
  dataset_downloader/
    faceforensics/
      download_faceforensics.py
    # future: celeb_df/, real_ai_videos/, etc. -- one subpackage per dataset
    # that actually has a scriptable downloader (see "genuinely missing" below)
  pre/
    pre_process_videos.py     # CPU frame extraction (unchanged by this reorg)
    pre_process_videos_gpu.py # GPU-accelerated variant (unchanged by this reorg)
    extract_faces.py           # face detect + align + crop + landmark sidecar (RetinaFace/insightface)
```

`manifest.py` and `integrity_scan.py` stay at the top level rather than
under a dataset/category subfolder because they're cross-cutting — every
dataset and every preprocessing step feeds into and reads from the same
manifest schema, so they don't belong to one category.

## What's here

- `dataset_downloader/faceforensics/download_faceforensics.py` — FaceForensics++
  downloader. Promoted from `temp/videos/faceforensics++/download.py` with
  one hardening fix: resume now requires the existing file to clear a
  minimum size threshold before being skipped, not just exist (an
  interrupted/truncated prior download used to be silently treated as
  complete). Not checksum-verified — FF++'s public download protocol
  doesn't publish per-file checksums.
  Run: `python -m proc.dataset_downloader.faceforensics.download_faceforensics <out_dir> -d Deepfakes -c c23 -t videos`
- `integrity_scan.py` — wraps `data/dataset_run.py`'s existing validation
  logic (not a rewrite) and adds the piece that was missing: a persisted,
  machine-readable manifest (`proc/manifest.py`'s schema) instead of only a
  console printout.
  Run: `python -m proc.integrity_scan --out proc/integrity_manifest.json`
- `manifest.py` — the manifest schema. `data/dataloader.py:DatasetBuilder`
  accepts an optional `manifest_path=` and will drop any file the manifest
  marked invalid at `build()` time — this is the "drop" decode-failure
  policy option T1.7 deferred here on purpose (dropping needs a
  pre-computed validity list; a per-`__getitem__` decision can't do it).
- `pre/pre_process_videos.py` / `pre/pre_process_videos_gpu.py` — existing
  frame extractors (contents unchanged by this reorg, only moved into
  `pre/`), auxiliary to the raw-video-first training path.
  Run: `python -m proc.pre.pre_process_videos ...`
- `pre/extract_faces.py` — face detection + alignment + crop + landmark
  extraction. Detector: RetinaFace via `insightface` (GPU-native,
  dense-enough landmarks for `docs/code/sbi.py`'s convex-hull blend mask,
  strong detection on compressed/off-angle faces). Records `det_score` per
  face. Cache format/validity: `data/cache.py` — packed format B
  (WebDataset-shard convention: size-bounded `.tar` shards, each sample a
  `<key>.jpg` crop + `<key>.json` metadata pair co-locating crop bytes,
  landmarks, det_score, video id, frame idx, and label in one record), plus
  a `cache_index.json` for cheap per-video cache-validity checks without
  scanning shards. Pure stdlib `tarfile` — no new dependency. Supersedes an
  earlier folder-per-video + `landmarks.json` design (format A) that shipped
  briefly; retired for the many-tiny-files problem packed formats exist to
  avoid at this corpus's scale (~12k videos × dozens of sampled frames each).
  **Caveat**: `insightface`/`onnxruntime` aren't installed in the sandbox
  this was written in (no GPU there), so the actual `FaceAnalysis.get()`
  detection call has not been exercised end-to-end — only the surrounding
  discovery/cache/crop-geometry logic was verified (against the real
  corpus and a synthetic cache round-trip suite). Smoke-test with `--limit 5`
  before a full run.
  Run: `python -m proc.pre.extract_faces --datasets celeb-df-v2 --limit 5 --frame-stride 8`

## What's genuinely missing, not just unbuilt

- **No downloader for `celeb-df-v2` or `real-ai-videos`.** Celeb-DF-v2's
  official distribution requires a request-access form from the dataset
  authors, not a public scriptable URL `[INFERRED from general public
  knowledge of that dataset's release process — not verified against any
  repo-internal evidence]`. `real-ai-videos`' origin isn't documented
  anywhere in this repo (`[UNKNOWN]`). Nothing to script here without
  knowing where that corpus actually came from. When/if either becomes
  scriptable, it gets its own `dataset_downloader/<name>/` subpackage,
  matching the `faceforensics/` pattern.
