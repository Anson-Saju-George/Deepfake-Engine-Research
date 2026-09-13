# FF++ Download — complete the 4 manipulations + DFD (c23). Paste into Claude Code (WSL)

## GOAL
Download the missing FF++ manipulations (and optionally re-fetch the existing ones for a
clean unified tree) using the confirmed-working local script. All c23 (matches disk).
Everything lands in ONE consistent FF++ directory structure.

## TARGET LOCATION
Download into a single clean FF++ root. Propose the path first and confirm with me — likely
F:/deepfake-video-ds/raw/faceforensics++ (raw video, NOT the crop cache), or the existing
FF++ root if we're merging in place. State which, and the resulting folder layout, before
downloading anything.

## PART A — REQUIRED: the 3 missing manipulations (~10GB, ~1000 videos each)
Using the local faceforensics_download_v4.py, c23, videos, EU2 server:
```
python <script> <FFPP_ROOT> -d Face2Face      -c c23 -t videos --server EU2
python <script> <FFPP_ROOT> -d FaceSwap       -c c23 -t videos --server EU2
python <script> <FFPP_ROOT> -d NeuralTextures -c c23 -t videos --server EU2
```

## PART B — OPTIONAL (ask me before running): re-fetch existing for a clean unified tree
Only if I confirm I want the clean-structure re-download (otherwise SKIP — existing data is
already c23 and integrity-verified):
```
python <script> <FFPP_ROOT> -d original                    -c c23 -t videos --server EU2
python <script> <FFPP_ROOT> -d Deepfakes                   -c c23 -t videos --server EU2
python <script> <FFPP_ROOT> -d DeepFakeDetection           -c c23 -t videos --server EU2
python <script> <FFPP_ROOT> -d DeepFakeDetection_original  -c c23 -t videos --server EU2
```
Do NOT run Part B without my explicit yes — it re-fetches ~14GB of data already on disk.

## SKIP
- FaceShifter (not part of standard FF++ four; not needed)
- c40 / raw (disk is c23 throughout; mixing compression would confound training)
- masks (-t masks) unless I ask

## RUN DISCIPLINE
- Run downloads in the background, logging to a file (this is a multi-GB, long pull).
- Use EU2 (CA server has known issues; fall back to EU if EU2 stalls).
- After each manipulation completes, report: video count landed, total size, and a quick
  ffprobe on 1 file to confirm valid h264 c23 video (not an HTML error page).
- If access fails partway (token/server), STOP and report the exact error.

## VERIFY AFTER DOWNLOAD (before we extract anything)
- Report final per-manipulation counts: original / Deepfakes / Face2Face / FaceSwap /
  NeuralTextures / DeepFakeDetection / DeepFakeDetection_original.
- Confirm F2F/FS/NT each ~1000 videos and share the same source-identity naming
  (000_003.mp4 -> id 000) as existing Deepfakes, so identity-disjoint splitting still works
  across all 4 manipulations. Flag any count shortfall.
- Report the new real:fake ratio across the full FF++ set (adding ~3000 fakes shifts it) —
  we'll recompute the clip-balance ratio from this before rebuilding the train cache.

## CONSTRAINTS
- c23 only. Videos only. Datasets read side is fine; write only to the FF++ root.
- No FaceShifter, no c40/raw.
- Part B needs explicit approval. Never print access tokens.

Propose the target path + folder layout first. Then run Part A. Ask before Part B.
