# Clip-Sampling Spec + seq={8,16} Benchmark — paste into Claude Code, run in WSL /opt/ml

## ROLE
Lock ONE clip-sampling recipe into the pipeline, verify it end-to-end on the real corpus,
then benchmark training throughput at seq_len=8 vs seq_len=16 so we pick a final config
with no decode bottleneck. Read data/cache.py, data/dataloader.py, extract_faces.py, and
CONVENTIONS.md FIRST; report current seq_len/stride/fps handling (path:line) before
changing anything.

## THE LOCKED RECIPE — do not alter these, they are decided
- **Cache at NATIVE fps.** Extraction stores one face crop per ORIGINAL video frame. No
  fps resampling, no frame-rate normalization anywhere.
- **stride = 1**, always, on the cached (native) frames. Consecutive frames. No stride
  variation, ever. Do not add adaptive/target-fps stride logic.
- **seq_len ∈ {8, 16}** — the ONLY variable under test.
- **Known + accepted property (state in CONVENTIONS.md, do not "fix"):** because fps is
  native and stride=1, a clip's real-time span varies with source fps
  (span = seq_len / src_fps). At 30fps: seq=8 → ~0.27s, seq=16 → ~0.53s. This per-video
  span variance is an accepted design choice and is documented as a limitation, NOT a bug
  to correct. Do not introduce resampling to "fix" it.
- **Short-clip policy:** if a video has fewer than seq_len frames, loop-pad (repeat from
  start) to seq_len. Never drop the sample. Count + log padded videos per dataset.
- **Multi-clip eval:** tile each video into NON-OVERLAPPING seq_len windows over its cached
  native frames, score each window, average to ONE video-level prediction. Report average
  clips/video (will be higher than usual since native-fps clips are short — expected).

If cache.py / extract_faces.py currently resample fps, REMOVE that so caching is native.
Flag it in the plan first.

## CONSTRAINTS
1. Run in /opt/ml (torch 2.11.0+cu129, cuDNN ON). Confirm before benchmarking; log
   cudnn.enabled==True and cv2.ocl.useOpenCL()==False.
2. No package installs/upgrades. Datasets read-only. Writes under test/ and the cache dir
   only.
3. AMP bf16 + channels_last for all throughput measurement (decided).
4. num_workers=12 (decided).
5. Label convention unchanged (real=1/fake=0, convert at metrics boundary).

## PLAN FIRST (propose, wait for approval before editing code or running full benchmark)
Report:
- current fps/stride/seq behaviour in cache.py + dataloader.py + extract_faces.py (path:line)
- confirmation the cache is (or will be made) native-fps with stride=1
- where seq_len is threaded (dataloader, registry, CLI)
- the loop-pad location and the multi-clip eval tiling location
- whether the cache already exists on disk; if not, this benchmark is BLOCKED on the
  extraction pass — say so and propose folding the two together

## VERIFICATION (correctness gates speed — before any throughput number)
On a 10-video sample spanning 25/30/60 fps:
- assert cached frames are the ORIGINAL consecutive frames (native fps, no resample):
  print src_fps, cached_frame_count, and confirm cached_count == original decodable count
- assert stride=1 sampling returns consecutive frame indices (0,1,2,... not 0,4,8,...)
- assert seq=16 span in seconds == 16 / src_fps per video (i.e. it CORRECTLY varies with
  fps — this documents the accepted variance; it must NOT be constant)
- assert loop-pad triggers on a < seq_len video
- assert non-overlapping tiling reuses no frame across windows
Print: video_id | src_fps | cached_frames | seq16_span_s (= 16/src_fps) | n_eval_clips.
seq16_span_s SHOULD differ across fps rows (that's the accepted native-fps behaviour).

## THROUGHPUT BENCHMARK (after verification passes)
ConvNeXt-Base, reading from the native-fps cache (NOT raw video):
- seq_len=8 and seq_len=16, stride=1
- report per seq_len: clips/sec, frames/sec, per-step latency (median + p95), peak VRAM,
  max stable batch size
- two conditions each: idle, and under a concurrent train loop (realistic)
- prove NO video decode in the training loop: report per-batch dataloader wait time (cache
  reads only). If dataloader wait is near-zero, the bottleneck is gone — state that.
- if seq=16 hits a VRAM/throughput wall, report the max batch that fits and clips/sec there

## OUTPUT → test/results/SAMPLING_BENCH.md
- the locked recipe stated once (native fps, stride=1, seq∈{8,16}, loop-pad, non-overlap
  eval, accepted span-variance limitation)
- verification table (proving native consecutive frames + fps-dependent span)
- seq=8 vs seq=16 throughput table (idle + under load, batch ceilings)
- one-line recommendation: seq_len, batch size, projected epoch wall-clock on full FF++
- any change made to cache.py / extract_faces.py / dataloader.py, with path:line

Report the plan first. No code edits or full benchmark until I approve.
