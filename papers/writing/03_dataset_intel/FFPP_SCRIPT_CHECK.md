# FF++ Download Script — locate + tiny test download. Paste into Claude Code (WSL)

## GOAL
Find the FaceForensics++ download script if it already exists locally, confirm it still
has working server access, by pulling ONE small file to a temp dir and then deleting it.
This verifies we can fetch the missing F2F/FS/NT manipulations WITHOUT waiting on a new
access form — before committing to the full ~10GB pull.

## STEP 1 — LOCATE THE SCRIPT (read-only search)
Search the whole system, not just the repo:
```
# common names across versions
find / -iname "faceforensics_download*.py" 2>/dev/null
find / -iname "*faceforensics*download*" 2>/dev/null
find /mnt/c /mnt/d /mnt/f ~ -iname "*faceforensics*.py" 2>/dev/null
```
Also check likely spots: the repo (proc/, datasets/, downloaders), ~/Downloads,
/mnt/c/Users/*/Downloads, and anywhere the existing DF/DFD videos live (the script was
probably run from near there originally).
Report every hit with full path + last-modified date. If multiple versions, prefer the
newest (v4 is latest). If NONE found, STOP and report "script not present — access form
required" — do not try to reconstruct or download the script from a fork.

## STEP 2 — INSPECT (don't run yet)
For the found script, report:
- its version / header comment
- `python <script> --help` output (args: dataset -d, compression -c, type -t, server)
- confirm it supports selective `-d Face2Face/FaceSwap/NeuralTextures` and `-c c23`
- whether it hardcodes a server URL / access token, and whether that looks intact
  (do NOT print any secret token value — just say present/absent)

## STEP 3 — TINY TEST DOWNLOAD (the real check, minimal footprint)
Goal: prove server access works by fetching the SMALLEST possible real payload, to temp.
```
mkdir -p /tmp/ffpp_test
```
Try, in order, the least-data option the script supports:
1. First choice: the script's built-in sample/benchmark flag if it has one
   (some versions: `-t videos --server EU2` with a `--num_videos 1` or similar). Check
   --help for any "number of videos" / "sample" limiter.
2. If a count limiter exists: download exactly 1 Face2Face c23 video to /tmp/ffpp_test.
3. If NO count limiter exists (script only does full-set): DO NOT run it — that would
   start a multi-GB pull. Instead report that a partial test isn't possible with this
   script version, and that the safest verification is to start the real F2F download
   with output to F: and kill it after the first file appears. Ask before doing that.

Use EU/EU2 server (repo notes the CA server has issues). Log: did it connect? did a real
video file land? file size + a quick ffprobe to confirm it's a valid playable video (not
an HTML error page — a common failure mode when access is expired).

## STEP 4 — VERIFY + CLEAN
- Confirm the downloaded file is a real video (ffprobe: has a video stream, sane duration)
  not a 403/HTML error page saved as .mp4.
- Report: SUCCESS (access works, ready for full F2F/FS/NT pull) or FAIL (with the exact
  error — expired token, dead server, script version too old).
- Delete /tmp/ffpp_test entirely. Confirm it's gone. Reclaim all test bytes.

## CONSTRAINTS
- Read-only search in Step 1. No downloads except the single tiny test file in Step 3.
- Never print access tokens/secrets.
- If the only option is a full-set download (no count limiter), STOP and ask — do not
  kick off a multi-GB pull to "test."
- Clean up /tmp/ffpp_test no matter what the outcome.

Report Step 1 (found or not) first. If found, proceed through 2-4 and give a clear
GO/NO-GO on whether we can pull F2F/FS/NT without a new access form.
