# Smoke Extract + seq 8-vs-16 Benchmark (one run) — paste into Claude Code, run in /opt/ml

## GOAL
Answer one question so training can begin: **does seq_len=8 and seq_len=16 work with good
throughput and no decode bottleneck, reading from the native-fps cache?** To answer it, we
must first create a small cache (the sample already selected in the previous step), then
benchmark seq 8 vs 16 against it. Both happen in this one /opt/ml session.

Scope note: this is a GO/NO-GO on the PIPELINE + PERFORMANCE, not the full corpus. A few
hundred cached clips is enough to measure throughput. Corpus-naming / LOMO / paper-scope
decisions are explicitly OUT OF SCOPE here — do not raise them.

## PART 1 — EXTRACT THE APPROVED SAMPLE (create the cache)
- Use the ~18-video stratified sample already selected (recover the dropped #15/#16 first;
  ensure at least one DeepFakeDetection real-paired case is included).
- Run extract_faces.py: native fps, stride=1 (NATIVE_FRAME_STRIDE), YuNet detector
  (confirm YuNet is wired; if still RetinaFace, confirm insightface/onnxruntime-gpu present
  before running). Store per crop: JPEG bytes, 5-pt landmarks, det_score, video_id,
  frame_idx, label.
- Log seconds/video and crops/video.

## PART 2 — QUICK CORRECTNESS GATE (must pass before benchmarking)
Fast checks only — full validation can come later, but these gate the benchmark:
- [ ] crops are 256px square, correct dtype, face roughly centered, ~1.3x margin
- [ ] 5 landmarks per crop, inside crop bounds, translated to crop space
- [ ] cache round-trip: read_keys() returns byte-identical crops
- [ ] a 60fps sample video yields ~2x the crops of a same-duration 30fps one (proves
      native-fps, not silently subsampled)
- [ ] loop-pad triggers on the 1-frame video (#4) without error
- [ ] SAVE one contact-sheet PNG (a few crops + landmark dots) and one SBI triptych PNG
      (real / blended / mask) using stored landmarks — for me to eyeball
If any of these fail, STOP and report — do not benchmark a broken cache.

## PART 3 — seq 8 vs 16 THROUGHPUT (the actual question)
Read ONLY from the cache built in Part 1 (never raw video). ConvNeXt-Base, AMP bf16,
channels_last, num_workers=12, batch=4. Confirm+log cudnn.enabled==True and
cv2.ocl.useOpenCL()==False first. Also try cudnn.benchmark=True (cheap, may lift numbers).

For seq_len=8 AND seq_len=16, report:
- clips/sec, frames/sec, per-step latency (median + p95)
- peak VRAM, and max stable batch size (do NOT exceed batch that fits — batch>=12 silently
  spills to system RAM per prior finding; cap and note it)
- **dataloader wait time per batch** — this is the bottleneck proof. Near-zero = cache
  reads aren't the bottleneck = we're good to train. Report it explicitly.
- run each idle AND under concurrent train load (realistic)

## PART 4 — VERDICT → test/results/SEQ_BENCH.md
- Part 2 pass/fail + paths to the two PNGs
- seq=8 vs seq=16 table (clips/sec, VRAM, batch ceiling, dataloader-wait, idle + under load)
- projected epoch wall-clock for a real training run at each seq_len (extrapolate from
  clips/sec and the balanced clip count)
- **clear GO / NO-GO**: is the pipeline + performance good enough to start the real
  training cycle? If GO, state recommended seq_len + batch + expected epoch time.
- recomputed full-corpus extraction wall-clock + disk (native fps, from THIS sample's real
  crops/video — expect ~4x the old stride-4 estimate)

## CONSTRAINTS
- /opt/ml only. Datasets read-only. Writes under test/ + scratch cache dir only.
- Only the ~18 approved videos. No full-corpus run.
- No package installs without flagging first (esp. pyav — likely NOT needed since training
  reads cache, not video; only extraction touches decode).
- If Part 2 fails, no benchmark, no GO.

Report Part 1+2 results and the two PNG paths first; if they look right I'll say continue
to Part 3, or you may proceed straight through if all Part 2 checks pass cleanly.
