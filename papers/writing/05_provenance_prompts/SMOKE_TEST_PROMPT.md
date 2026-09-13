# Extraction Smoke Test — stratified sample, then validate. Paste into Claude Code, run in /opt/ml

## ROLE
Before committing to a full ~4h native-fps extraction over 12,028 videos, prove
extract_faces.py produces correct crops + landmarks + a usable SBI blend on a SMALL,
DELIBERATELY STRATIFIED sample that spans the real corpus distribution — not the 5
easiest videos. Select the sample from a distribution analysis FIRST, extract, then
validate against explicit pass/fail criteria. Report and STOP — no full extraction until
I approve.

## PHASE 0 — CORPUS DISTRIBUTION ANALYSIS (choose the sample from data, not by guessing)
Probe the real corpus (ffprobe/cv2, read-only) and report distributions for:
- fps: min / max / mean / median / mode, + histogram buckets (15/24/25/30/50/60)
- resolution (short side): buckets <=320 / 321-480 / 481-720 / >720
- duration (s) and frame_count: min / max / mean / median
- codec: mpeg4 / h264 / vp9 / other counts
- dataset: celeb-df-v2 / faceforensics++ / real-ai-videos
- per FF++ manipulation: Deepfakes / Face2Face / FaceSwap / NeuralTextures (+ real)
Note mean-vs-median gaps (skew) and the mode fps — the sample must cover the mode AND
the tails, since native-fps stride=1 behaves differently at 60fps vs 30fps.

## PHASE 1 — SELECT THE SAMPLE (propose the list, wait for my OK before extracting)
Pick ~15-20 videos (small enough to eyeball every one) that DELIBERATELY span:
- each dataset (celeb-df-v2, faceforensics++, real-ai-videos)
- each FF++ manipulation type + at least one FF++ real
- both label classes (real and fake)
- fps: at least one each at the mode (~30) AND both tails (min ~15 and max ~60) — the
  60fps case is critical (native-fps stride=1 → shortest real-time span)
- a resolution low-end and high-end
- these EDGE CASES explicitly (pull from the corpus profile / bad_video_log if present):
  * the shortest-duration video (the 0.033s outlier flagged earlier, if it exists)
  * a very long video (near max duration)
  * a known-difficult / previously-flagged decode video, if any
  * ideally one likely low-face-rate video (profile/occlusion) to exercise the gap logic
Output a table: video_id | dataset | manip | label | fps | resolution | duration | why_selected.
Wait for my approval of this list before Phase 2.

## PHASE 2 — EXTRACT (only the approved sample)
Run extract_faces.py on the sample, native fps, stride=1 (NATIVE_FRAME_STRIDE), YuNet
detector (confirm it's the wired detector; if it's still RetinaFace, say so and confirm
insightface/onnxruntime-gpu presence before running). Store per crop: JPEG bytes,
5-point landmarks, det_score, video_id, frame_idx, label. Log timing per video.

## PHASE 3 — VALIDATE (explicit pass/fail; a fail blocks the full run)

### A. Crop geometry
- [ ] crop size == configured (256), square, correct dtype/uint8, BGR or RGB stated
- [ ] ~1.3x margin present (face not edge-to-edge; blend-boundary region included)
- [ ] face roughly centered; report mean/median face-center offset
- [ ] SAVE a contact sheet: 3-5 crops per video as a grid PNG for me to eyeball

### B. Landmarks (SBI depends on these)
- [ ] 5 landmarks per crop, coordinates INSIDE crop bounds (0..255)
- [ ] landmarks translated into crop space (not left in original-frame coords)
- [ ] overlay landmarks on a few crops in the contact sheet (dots on eyes/nose/mouth)
- [ ] det_score stored; report min/mean/median; flag crops with det_score < 0.6

### C. Face-found rate + temporal gaps (the P5 check, run here early)
Per video report: original_decodable_frames | cached_frames | face_found_rate |
max_frame_idx_gap. Flag face_found_rate < 0.9 or max_frame_idx_gap > 5. Report the
worst offenders — these quantify the detection-gap span-variance limitation.

### D. SBI smoke (the real point of RetinaFace-vs-YuNet)
- [ ] run dfx/sbi.py self_blend on 3-5 real crops using the STORED landmarks
- [ ] SAVE before/after/mask triptychs to a PNG for me to see
- [ ] confirm the 5-point convex-hull mask covers the face plausibly (not a tiny blob,
      not the whole frame). If YuNet's 5 points give a poor mask, SAY SO — that's the
      signal to reconsider RetinaFace. This is the decision this smoke test exists to make.

### E. Short-clip + native-fps sanity
- [ ] loop-pad triggers correctly on any sample video with < seq_len cached frames
- [ ] a 60fps video yields ~2x the cached frames of a same-duration 30fps video
      (proves native-fps really is native, not silently subsampled)
- [ ] cache round-trips: read_keys() returns byte-identical crops to what was written

### F. Cache integrity
- [ ] shard(s) + index written, index entry count == crops written
- [ ] no orphan keys, no duplicate keys
- [ ] projected full-corpus size + wall-clock, recomputed from THIS sample's real
      crops/video and seconds/video (not the old stride-4 estimate)

## OUTPUT → test/results/SMOKE_TEST.md
- Phase 0 distribution tables (with mean/median/mode + skew notes)
- Phase 1 selected-sample table
- Phase 3 pass/fail checklist, every item marked PASS/FAIL/[NEEDS EYE]
- paths to the contact-sheet PNGs and SBI triptych PNGs (I will look at these)
- the recomputed full-corpus extraction projection (wall-clock + disk) at native fps
- a clear GO / NO-GO recommendation for the full extraction, with the detector decision
  (YuNet confirmed, or switch to RetinaFace) justified by the SBI mask result

## CONSTRAINTS
- Run in /opt/ml. Datasets read-only. Writes under test/ and a scratch cache dir only.
- Only extract the ~15-20 approved videos. No full-corpus run.
- If a validation item FAILS, stop and report — do not proceed to a GO recommendation.
- Propose the Phase 1 sample list and wait for my approval before extracting.

Start with Phase 0, report the distributions + proposed sample list, and wait.
