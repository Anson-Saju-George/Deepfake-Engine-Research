# Performance — Consolidated Summary

Plain-language rollup of everything tested under `test/` (Tracks A-E: CPU/GPU decode, DataLoader tuning, NVDEC, Intel iGPU, face detection, training throughput) plus the CUDA/cuDNN environment check. Full detail with raw numbers and methodology lives in `test/results/` — this doc is the "what do I actually do" summary.

**Three source documents, read in this order of authority:**
1. `test/results/CUDA_VERIFY.md` — confirms `/opt/ml` (the real training venv) has a *healthy* cuDNN, unlike the disposable benchmark venv used for the first pass.
2. `test/results/REPORT_v2.md` — re-run of the training-throughput numbers in `/opt/ml` with cuDNN on. **Supersedes REPORT.md's Track E numbers.**
3. `test/results/REPORT.md` — the original, larger run (decode backends, workers, NVDEC, Intel, face detection, first-pass training numbers). Still authoritative for everything `REPORT_v2.md` didn't re-test (decode backends aren't cuDNN-sensitive, so those numbers stand).

Not covered here: the native-fps cache-read benchmark (`seq_len` 8 vs 16, reading from `data/cache.py`'s packed cache instead of raw video) — that's a separate, not-yet-run benchmark, will land in `test/results/SAMPLING_BENCH.md` once it's done.

---

## Bottom line — what to actually set

| Setting | Value | Why |
|---|---|---|
| CPU decode backend | `pyav` | ~7.5x faster than `cv2`, pixel-identical output at the target clip shape. **Caveat: not installed in `/opt/ml` today** — see "Gaps" below. |
| GPU/NVDEC decode | **Don't use it** | Every GPU decode path tested was slower than CPU `pyav`, and two of them (`torchcodec_cuda`, `ffmpeg_cuda_cli`) had real correctness bugs on specific videos. |
| Intel iGPU (decode or compute) | **Don't use it** | Every path fails — `/dev/dri` doesn't exist in this WSL2 setup. Platform limitation, not fixable in software. |
| DataLoader `num_workers` | **12** | Throughput plateaus there; more workers add contention risk with no benefit. |
| Face detector (extraction pass) | `YuNet` (CPU, ships inside `opencv-python`) beats `MTCNN` (CUDA) | ~7.7x faster in this environment, and simpler — no extra dependency, unlike MTCNN's `facenet-pytorch` (which isn't even installed in `/opt/ml`). **This does NOT settle RetinaFace vs YuNet** — RetinaFace (what `extract_faces.py` actually uses today) was never in this comparison. Still an open call. |
| AMP dtype | **bf16** | Within 3.5% of fp16's speed, safer for this corpus's compressed/dark frames (no loss-scaling needed). +56% throughput over fp32, ~42% less VRAM. |
| `channels_last` | Only combined with AMP | Alone (fp32), it's actually a regression with cuDNN on (0.67 vs 3.25 clips/sec) — a real, reproducible finding, likely fixable with `cudnn.benchmark=True` (not yet tried). Combined with AMP bf16 it's fine (13.46 vs bf16-alone's 13.43). |
| `torch.compile` | **Don't use it** | Still ~11x slower than plain AMP even after the cuDNN fix. |
| Batch size | **4** (clean, 11.5 clips/sec, 8.5GB) | 8 works but throughput collapses (~1.9 clips/sec) as VRAM nears the card's 16.3GB ceiling. **Never go to 12+** — the system silently spills into system RAM instead of failing cleanly, which is worse than a crash. |

---

## 1. Environment — which one actually matters

Two separate environments were involved, and mixing up their numbers is the main way to misread these reports:

- **The disposable benchmark venv** (`test/.venv`, since deleted) — `torch==2.13.0+cu132`. This venv had a real, environment-specific bug: enabling cuDNN crashed ordinary `conv2d` calls (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`). Every number from the *first* benchmark pass (`REPORT.md`'s Track E training numbers, and its MTCNN face-detection number) was measured with cuDNN **forcibly disabled** as a workaround — those numbers are a floor, not a ceiling.
- **`/opt/ml`** — the real, actual training venv (`torch==2.11.0+cu129`), confirmed healthy: cuDNN works normally, `conv2d` in fp32, bf16 autocast, and a full ConvNeXt-Base forward+backward pass all ran cleanly (`CUDA_VERIFY.md`). **This is the environment whose numbers should be trusted for training decisions.**

Hardware for both: RTX 5080 Laptop GPU (16.3GB VRAM, `sm_120`/Blackwell), Intel Core Ultra 7 255HX (20 logical CPUs), WSL2 Ubuntu.

---

## 2. Decode backend — CPU vs GPU

Tested against the real 12,028-video corpus (celeb-df-v2, faceforensics++, real-ai-videos — fps ranges 15-60, median 30).

**Correctness first**: at the target clip shape (16 frames, stride 4), `cv2`, `pyav`, and `decord` (CPU) all produce pixel-identical output. Both GPU decode candidates had real problems — `torchcodec_cuda` showed one video with a full-value pixel mismatch (not just rounding drift), and `ffmpeg_cuda_cli` outright failed to retrieve specific requested frames on 2 videos. Neither is a safe drop-in replacement without per-video validation.

**Speed** (target clip shape, warm cache, idle):

| Backend | clips/sec | vs `cv2` |
|---|---|---|
| `cv2` (baseline) | 0.99 | 1.0x |
| `decord` (CPU) | 4.2 | 4.2x |
| **`pyav` (CPU)** | **7.4** | **7.5x** |
| `torchcodec` (GPU/NVDEC) | 3.1 | 3.2x — slower than CPU `pyav` |
| `ffmpeg_cuda_cli` (GPU) | too slow to be usable (~0.8s per clip, spawns a fresh process every time) | — |

**Verdict: `pyav` on CPU wins outright.** GPU/NVDEC decode isn't worth pursuing for this corpus — it's both slower and riskier.

---

## 3. DataLoader worker count

Simple sweep, `cv2` backend, target clip shape:

| workers | clips/sec |
|---|---|
| 0 | 1.32 |
| 4 | 2.29 |
| 8 | 2.30 |
| **12** | **2.53** |
| 16 | 2.53 |
| 18 | 2.48 |

Flat past 12 (even a slight dip at 18) — **cap at 12** on this 20-core machine.

---

## 4. Intel iGPU — dead end, confirmed twice

Every path tried — VA-API, QSV, native `torch.xpu`, Intel's IPEX extension, OpenVINO's GPU plugin — fails for the **same single reason**: `/dev/dri` (the Linux device node all of these need) doesn't exist in this WSL2 instance. Re-confirmed in the `/opt/ml` rerun with identical failures. This is a platform limitation (WSL2's virtualized GPU passthrough), not something any software fix here can solve. IPEX is also separately end-of-life as of March 2026, moot regardless.

**Verdict: don't spend more time on Intel iGPU in this environment.**

---

## 5. Face detection — extraction-pass speed

| Detector | frames/sec | Notes |
|---|---|---|
| MTCNN (CUDA) | 20.5 | Needs `facenet-pytorch`, not installed in `/opt/ml`; number is from the cuDNN-disabled venv, likely understated |
| **YuNet (CPU)** | **103-158** (varies by run/venv) | Ships inside `opencv-python`, already a hard dependency — zero extra install |

YuNet wins on both speed (~5-7.7x) and simplicity (no new dependency). **Important caveat**: this comparison never included RetinaFace, which is what `proc/pre/extract_faces.py` actually uses today — that choice was made earlier for its landmark quality (needed for SBI augmentation) and hasn't been re-evaluated against YuNet's speed. Still an open question, not resolved by this benchmark.

---

## 6. Training throughput — the core numbers, `/opt/ml` cuDNN-on

ConvNeXt-Base, batch=4 clips unless noted:

| Config | clips/sec | Peak VRAM |
|---|---|---|
| fp32, default layout | 3.25 | 14.6GB |
| fp32, `channels_last` | 0.67 (regression — see below) | 14.6GB |
| AMP fp16 | 13.90 | 8.5GB |
| **AMP bf16** | **13.43** | **8.5GB** |
| AMP bf16 + `channels_last` | 13.46 | 8.5GB |

**AMP (bf16) is the single biggest lever**: +56% throughput, ~42% less VRAM, versus plain fp32. Use it unconditionally.

**`channels_last` alone is a real regression** with cuDNN enabled (not an artifact — reproduced cleanly): cuDNN's default algorithm-selection heuristic seems to mis-pick for this layout without `cudnn.benchmark=True` enabled (that flag wasn't tried yet — flagged as a cheap follow-up). Only use `channels_last` combined with AMP, where it's fine.

**Batch size** — AMP-specific sweep:

| Batch | clips/sec | VRAM |
|---|---|---|
| **4** | **11.5** | **8.5GB — clean** |
| 8 | 1.9 (steep drop) | 16.4GB — essentially the card's full capacity |
| 12 | 0.56 | 24.3GB reported (**exceeds physical VRAM** — silent spill into system RAM) |
| 16 | 0.41 | 32.1GB reported (same silent-spill problem, worse) |

**Batch=4 is the sweet spot.** Batch=8 is usable only if you specifically need it, at ~6x lower throughput. **Never go to 12+** — it doesn't fail loudly, it just gets much slower while claiming to still be running, which is a worse trap than a clean out-of-memory crash.

**`torch.compile`**: even after the cuDNN fix, steady-state throughput is ~1.3 clips/sec — still ~11x slower than plain AMP. Not worth it here, likely due to this GPU's SM count interacting badly with the compiler's autotuning ("Not enough SMs to use max_autotune_gemm mode" warning, unchanged with cuDNN on or off).

---

## 7. Gaps — what these numbers don't cover

- **`pyav` isn't installed in `/opt/ml`.** The decode-backend recommendation (§2) is correct but not actionable without installing a package — extraction and training currently still read via `cv2`. Installing `pyav` would need explicit sign-off (package-mutation decisions aren't made unilaterally in this project).
- **MTCNN and `torchcodec` couldn't be re-measured in `/opt/ml`** — neither package is installed there, and installing them to get a "real" number would have broken the zero-mutation rule for that rerun. Their numbers above are carried over from the disposable venv and may not hold exactly, though there's no specific reason to expect a large shift for either (decode/detection speed isn't strongly cuDNN-sensitive).
- **RetinaFace was never benchmarked against YuNet or MTCNN** — see §5. `extract_faces.py`'s actual detector choice is still unverified against this data.
- **`cudnn.benchmark=True`** wasn't tried — likely fixes the `channels_last` regression and could improve several other numbers further. Proposed, not tested.
- **Gradient checkpointing** wasn't measured — flagged as the natural follow-up if a larger effective batch size is ever needed to escape the batch=8+ VRAM cliff.
- **This whole document predates the native-fps cache rework** (`data/cache.py`/`data/cache_dataset.py`, packed WebDataset-shard cache, `stride=1` always, `seq_len` 8 vs 16). None of the decode-backend numbers above apply once training reads JPEG crops from the cache instead of raw video — that's a separate benchmark, still pending a real extraction pass in `/opt/ml`.
