# Cache Performance — celeb-df-v2, native-fps face-crop cache

Measured, not estimated. Full detail and methodology: `test/results/CACHE_LOCATION.md`. This file is the distilled numbers-only record.

## Extraction (one-time cost)

| | |
|---|---|
| Videos | 6,533 / 6,533 (100%, 0 failed) |
| Crops written | 2,470,703 (avg 378.2/video) |
| Face-found rate | 99.77% |
| Cache size | 30.36 GB (q90 JPEG, 256×256) |
| Detector | YuNet, resize-before-detect capped at 1,600px long side |

## Read throughput — full 2.47M-crop cache, no GPU

| Location | Filesystem | Cold crops/sec | Cold MB/sec | Warm crops/sec | Warm MB/sec |
|---|---|---:|---:|---:|---:|
| `F:` (T7 Shield, external) | 9p/drvfs | 820.4 | 7.85 | 2,527.9 | 24.19 |
| `D:` (internal) | 9p/drvfs | 1,383.4 | 13.24 | 2,980.2 | 28.51 |
| `~/dfcache` (WSL native) | ext4 | 19,319.3 | 184.84 | 48,021.5 | 459.46 |

ext4 is ~19x faster than `F:` warm, ~16x faster than `D:` warm. A shard-index sidecar fix (see `data/cache.py`) gave `F:` a 3.08x warm speedup over its own cold pass (820.4→2,527.9 crops/sec) by eliminating repeated `tarfile.getmembers()` scans.

## Read throughput under real GPU training load — the number that actually decides cache location

ConvNeXt-Base, AMP bf16, channels_last, batch=4, `num_workers=12`, reading from `F:` (the slower drvfs candidate — worst case tested).

| seq_len | clips/sec | dataloader wait (avg) | wait % of step | GPU util | peak VRAM |
|---|---:|---:|---:|---:|---:|
| 8 | 25.92 | 0.25 ms | 0.16% | 76.1% | 5.08 GB |
| 16 | 13.56 | 0.27 ms | 0.09% | 86.1% | 9.04 GB |
| **32** | **0.81** | 0.22 ms | 0.004% | 98.1%¹ | **17.26 GB²** |

Dataloader wait is ~0 at every seq_len tested, including 32 — disk is never the bottleneck, at any temporal length. seq 8/16 are GPU-bound (healthy). **seq 32 is not**:

¹ 98.1% "GPU util" at seq32 is misleading — this is memory-pressure thrashing, not healthy saturation (compare the 16.7x throughput collapse below).
² **Peak VRAM at seq32/batch4 (17.26GB) exceeds the card's 16.3GB physical capacity.** It did not fail with a clean OOM — it silently spilled past physical VRAM (the same WDDM/driver paging behavior documented in `REPORT_v2.md` for batch≥12, here triggered via `seq_len` instead of batch). Throughput collapsed **16.7x** (13.56→0.81 clips/sec; median step latency 0.295s→4.93s).

**Root-cause check — was this the DataLoader's 12 workers, not the model?** No. Workers are CPU-only (JPEG decode + transform in `data/cache_dataset.py`, no CUDA calls) and structurally cannot hold GPU memory — `torch.cuda.max_memory_allocated()` only tracks PyTorch's CUDA allocator, which workers never touch. Confirmed empirically: reran seq32/batch4 with `--num-workers 1` — **peak VRAM was `17258.1845703125` MB, identical to the workers=12 run to the decimal place.** The overflow is purely `batch_size × seq_len` (128 images/step at batch4/seq32) exceeding available activation memory; worker count is irrelevant.

**Lighter config found: batch=2, seq_len=32.** Same total images/step as batch4/seq16 (2×32 = 4×16 = 64) — and sure enough, peak VRAM (`9252.2470703125` MB) matched batch4/seq16's VRAM to the decimal place. Clean, GPU-bound, no overflow:

| Config | clips/sec | dataloader wait | wait % of step | GPU util | peak VRAM |
|---|---:|---:|---:|---:|---:|
| seq32, batch=4, workers=12 | 0.81 | 0.22ms | 0.004% | 98.1%¹ | 17.26 GB (overflow) |
| seq32, batch=4, workers=1 | 0.62 | 0.26ms | 0.004% | 99.3%¹ | 17.26 GB (overflow, confirms workers not the cause) |
| **seq32, batch=2, workers=12** | **6.71** | 0.26ms | 0.09% | 86.2% | **9.04 GB (fits, healthy)** |

**seq_len=32 is usable — at batch=2, not batch=4.** If more temporal context than seq16 is ever needed, this is the config to use.

## Epoch wall-clock projection, celeb-df-v2 (6,533 clips/epoch)

| Config | seconds/epoch | minutes/epoch |
|---|---:|---:|
| seq8, batch4 | 252 | 4.2 |
| seq16, batch4 | 482 | 8.0 |
| seq32, batch4 | 8,053 | 134.2 (not usable — VRAM overflow) |
| seq32, batch2 | 973 | 16.2 |

## Verdict

**Cache location: `F:`.** GPU-bound at every usable config tested, on the slower of the two drvfs candidates — no need for a faster ext4 staging copy.
**seq_len: 16 at batch=4** (recommended default) — 2x temporal context over seq8, still comfortably GPU-bound, VRAM well under the card's ceiling, best clips/sec-to-context tradeoff.
**seq_len=32 requires batch=2**, not batch=4 (confirmed not a worker-count issue) — usable if more temporal context is needed later, at ~2x the epoch time of seq16.
