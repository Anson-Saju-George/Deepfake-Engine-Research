# Performance Context — how to read `PERFORMANCE_LOG.md`

Context for the performance numbers only. **What the benchmarks were, the exact environment
they ran in, and — critically — what was and was NOT measured**, so nothing in
`perf/PERFORMANCE_LOG.md` is misread now that the `test/` benchmark harness has been deleted.
For whole-project scope (research direction, datasets, layout) see the root
`PROJECT_CONTEXT.md`.

---

## 1. What was benchmarked, and why

A 5-track sweep (A–E) plus a cache study, run to lock the training pipeline's performance
knobs before committing to large-scale runs:

| Track | Question it answered |
|---|---|
| **A** | Fastest CPU video-decode backend + optimal DataLoader worker count |
| **B** | Is any GPU/NVDEC decode path worth it? |
| **C** | Is there any usable Intel-iGPU acceleration in this WSL2 setup? |
| **D** | Which face detector (YuNet vs MTCNN) for crop extraction? |
| **E** | Best training precision / layout / `torch.compile` / batch size |
| **Cache** | Where to store the crop cache, and does disk speed bottleneck training? |

The distilled winners feed the "winning configuration" table in `PERFORMANCE_LOG.md §1`.

---

## 2. The environment the numbers came from

**Benchmarks (Tracks A–E) were certified in the production WSL venv `/opt/ml`** — torch
2.11.0+cu129, cuDNN 91701, cv2 4.13.0, on an RTX 5080 Laptop (16.3 GB, driver 610.47) under
WSL2 Ubuntu 24.04.4. The GPU numbers are the **cuDNN-ON** rerun (authoritative); an earlier
cuDNN-OFF pass is the "floor" and is only referenced where a number moved.

The 22 completed *training runs* used a **different** env — Windows conda `torch` (torch
2.10.0+cu130, CUDA 13.0). Full version lists for both are in `PERFORMANCE_LOG.md §1`. When
comparing a training-run number to a benchmark number, remember they may come from different
torch/CUDA builds.

---

## 3. ✅ What WAS measured

- Training throughput across fp32 / AMP fp16 / bf16, `channels_last`, `torch.compile`, batch 1–16 (Track E).
- CPU decode backends cv2 / decord / pyav across seq/stride, worker sweep 1–20 (Track A).
- Cache read throughput cold/warm on F:, D:, ext4, and **under real GPU training load** at seq 8/16/32 (Cache study).
- YuNet (CPU) face detection end-to-end (Track D).
- Model complexity — params/FLOPs/latency/throughput/memory for every registered architecture (`PERFORMANCE_LOG.md §4`).
- Full celeb-df-v2 crop-cache extraction (6,533 videos, 2.47M crops, 99.77% face-found).

## 4. ❌ What was NOT measured (do not cite as real)

- **MTCNN-CUDA with cuDNN ON** — `facenet-pytorch` absent from `/opt/ml`. Only a cuDNN-OFF number (20.53 fps) exists. Moot: YuNet was chosen.
- **`torchcodec_cuda` / NVDEC session sweep (B5)** — `torchcodec` absent. Raw "973–1026 clips/sec" values are the speed of an instant `ModuleNotFoundError`, **not measurements**.
- **`pyav` in `/opt/ml`** — `av` absent there; its 7.39 clips/sec is from the original disposable benchmark venv (not cuDNN-sensitive, so it stands, but wasn't re-run in prod).
- **Any Intel-GPU path** — VA-API / QSV / `torch.xpu` / OpenVINO / IPEX all probed, **all failed**.
- **`cudnn.benchmark=True`** — never enabled; it's the proposed fix for the `channels_last` regression, out of scope for the rerun.
- **Gradient checkpointing** — never measured; flagged as the escape from the batch≥8 VRAM cliff.

## 5. ⚠️ Caveats that change how the numbers read

- **The VRAM cliff is silent.** Past ~16.3 GB (batch≥12, or seq32/batch4) the WDDM/WSL2 driver spills GPU memory into system RAM and throughput collapses ~16× **without** raising `OutOfMemoryError`. This is why the recommended ceilings look conservative.
- **`channels_last` regression is real, not noise** — reproducible; caused by cuDNN's default heuristic mis-selecting NHWC without `cudnn.benchmark=True`.
- **fp16 ≈ bf16** (3.5% gap = noise); bf16 is recommended for NaN-safety on dark/compressed frames, not for speed.
- Benchmarks were run under the "zero package-mutation" constraint on `/opt/ml` — that is *why* several exploratory backends show as NOT re-run rather than as failures.

---

## 6. Files in `perf/`

| File | Contents |
|---|---|
| `PERFORMANCE_LOG.md` | **all numbers** — env, Tracks A–E, cache throughput, model complexity |
| `PERFORMANCE_CONTEXT.md` | this file — scope, environment, tested/not-tested, caveats |
| `CACHE_PERFORMANCE.md` | original distilled cache-study record (also folded into the log §3) |
