# FF++ — what's on disk vs what the download script offers. Paste into Claude Code (WSL)

## GOAL (read-only, no downloads)
Map exactly which FF++ variants I already have vs which the local download script can
fetch, so we know precisely what to pull.

## STEP 1 — WHAT'S ON DISK
List the FF++ manipulation folders actually present, with per-folder video counts:
```
find datasets/videos/faceforensics++ -maxdepth 4 -type d 2>/dev/null
```
Report which of these exist and how many videos each has:
- real / original
- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures
- FaceShifter
- DeepFakeDetection (DFD)
Note the compression level present (c0/raw, c23/HQ, c40/LQ) — infer from paths or ffprobe
bitrate on a sample. Report which manipulations are MISSING.

## STEP 2 — WHAT THE SCRIPT OFFERS
Locate the download script (faceforensics_download*.py — search /mnt/c /mnt/d /mnt/f ~ and
the repo). Run `python <script> --help` and report the full menu of choices it supports:
- dataset types (-d): all manipulation names it accepts
- compression (-c): c0 / c23 / c40
- file types (-t): videos / masks / models / etc.
- server options, and any per-video count / sample limiter flag
Do NOT download anything. Do NOT print tokens.

## STEP 3 — THE GAP TABLE
One table: variant | on-disk? | count | script can fetch it? | c23 available?
Then state plainly: which variants to download to complete standard FF++ (the 4:
Deepfakes/Face2Face/FaceSwap/NeuralTextures at c23), and whether FaceShifter/DFD are
extras already covered.

Read-only. Report the three sections. No downloads, no secrets printed.
