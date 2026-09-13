# Cache Location Test — F: (drvfs) vs native ext4. Paste into Claude Code, run in /opt/ml (WSL)

## GOAL
Decide where the training cache should live by MEASURING, not assuming. Extract ONE
dataset (celeb-df-v2) once, place a copy on each candidate location, and benchmark read
throughput + read-under-GPU-load on both. The cache is WebDataset tar-shards (few big
files, sequential reads) — the drvfs penalty is worst for many-small-files, so tar shards
MAY largely dodge it. That is the hypothesis to test.

## LOCATIONS UNDER TEST
- **F: (T7 Shield)** → /mnt/f/deepfake-video-ds — 9p/drvfs, external 2TB SSD, PERSISTS,
  visible to Windows, survives WSL reset. Intended production location.
- **native ext4** → e.g. ~/dfcache (on / , the WSL VM's own ext4 disk) — true native SSD
  speed, but trapped inside the WSL distro (lost if distro is reset, invisible to Windows).

Source videos are on /mnt/d (drvfs) regardless — extraction READ is bridge-bound either
way. That is a one-time cost, NOT the per-epoch training cost being optimized. Do not let
a slow extraction pass be misread as a slow cache. Only the CACHE location affects training
speed.

## PRE-CHECK
Confirm and print, for each location: `df -T <path>` and `mount | grep` — report filesystem
type (expect 9p for /mnt/f, ext4 for ~/dfcache). Report free space on each.

## STEP 1 — EXTRACT celeb-df-v2 ONCE (to F:)
Apply the agreed fixes first (confirm both in): resize-before-detect, JPEG q90.
Extract celeb-df-v2 (6,533 videos, ~2.48M frames), native fps, stride=1, YuNet, 256x256,
q90, writing shards to /mnt/f/deepfake-video-ds. Log: wall-clock, crops/video, final size
(expect ~26GB at q90 vs 34.77GB projection at q95). Note: extraction read is drvfs-bound
(source on D:) — report extraction time but do NOT treat it as the cache-speed signal.

## STEP 2 — MIRROR TO ext4
Copy the produced shards + index to ~/dfcache (native ext4). Verify byte-identical
(size + a checksum on the shard files). This is a one-time ~26GB local copy.

## STEP 3 — READ THROUGHPUT, BOTH LOCATIONS (no model)
For EACH of /mnt/f and ~/dfcache, iterate the ENTIRE celeb-df cache through the real cache
reader (data/cache_dataset.py path), no GPU, cold and warm:
- cold: first pass after `sync` (drop caches if sudo available; else note method)
- warm: second pass
Report per location: crops/sec, MB/sec sustained, cold vs warm. This is the raw disk
ceiling per filesystem.

## STEP 4 — READ UNDER GPU TRAINING LOAD, BOTH LOCATIONS (the real test)
For EACH location, run ConvNeXt-B, bf16, channels_last, batch=4, num_workers=12, reading
clips from that location's cache while the GPU actually trains. Report per location:
- clips/sec (training throughput)
- **dataloader wait time per batch** — THE bottleneck proof (near zero = disk feeds GPU)
- GPU utilization % (nvidia-smi dmon or `nvidia-smi --query-gpu=utilization.gpu`) — is the
  GPU saturated or starved?
- peak VRAM
Run seq_len=8 and seq_len=16 (the pending seq question rides along here for free).
Confirm+log cudnn.enabled==True, cv2.ocl.useOpenCL()==False.

## STEP 5 — VERDICT → test/results/CACHE_LOCATION.md
- pre-check table (fs type, free space per location)
- extraction result (time, size, crops/video) with the "extraction-read-is-drvfs" caveat
- Step 3 read table: /mnt/f vs ext4, cold+warm, crops/sec + MB/sec
- Step 4 table: /mnt/f vs ext4 x seq{8,16} — clips/sec, dataloader-wait, GPU-util, VRAM
- **the delta**: how much slower is F: (drvfs) than ext4 UNDER GPU LOAD? State it as a %.
- **recommendation**, decided by these rules:
  * If F: dataloader-wait is near zero AND GPU-util is high (GPU is the bottleneck, not
    disk) → USE F:. It persists, survives WSL reset, visible to Windows, and the tar-shard
    design absorbed the bridge penalty. This is the preferred outcome.
  * If F: shows high dataloader-wait / starved GPU but ext4 does not → the bridge bites;
    weigh ext4's speed against its fragility (lost on WSL reset, invisible to Windows) and
    recommend, noting the tradeoff explicitly.
  * If BOTH starve the GPU → the bottleneck isn't the filesystem; diagnose the reader
    before scaling.
- also settle seq_len: 8 vs 16 recommended config from Step 4, with epoch wall-clock
  projection for full-corpus training at the chosen location.

## CONSTRAINTS
- /opt/ml. Datasets read-only. Cache writes to /mnt/f/deepfake-video-ds and ~/dfcache only.
- Only celeb-df-v2. No full-corpus extraction yet.
- No package installs without flagging (pyav not needed — training reads cache, not video).
- Report pre-check + extraction size first; if extraction size or crops/video look wrong
  vs projection, STOP and report before the throughput tests.

Start with the pre-check, then extract celeb-df-v2 to F:, then run Steps 2-5.
