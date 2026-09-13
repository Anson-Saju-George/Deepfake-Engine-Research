# GPU Acceleration Benchmark Prompt — paste into Claude Code at the repo root

---

## ROLE

You are a **performance engineer running a controlled hardware evaluation**. Your job
is to answer one question with evidence:

> Which decode / dataloading / compute path gives the highest sustained training
> throughput at `seq_len=16, stride=4` on this machine — and does any of it change
> the pixels the model sees?

This is an evaluation, not an integration. Nothing you write here ships. You produce
measurements and a recommendation; I decide what gets adopted into the real pipeline
later.

**A fast decoder that returns different pixels than the current one is a correctness
bug, not an optimisation.** Throughput numbers are worthless without the frame-equivalence
check. Treat that as the primary gate.

---

## HARD CONSTRAINTS — SANDBOX

1. **All code, configs, results, and dependencies live under `test/`.** Not one byte
   changes anywhere else in the repo.
2. **Two isolated environments, both under `test/`.** `test/.venv` from
   `test/requirements.txt` (CUDA torch, Tracks A/B/D/E). `test/.venv-intel` from
   `test/requirements-intel.txt` (IPEX track only — it pins incompatible torch builds).
   Do not touch the WSL system environment or the existing training venv.
3. **Datasets are READ-ONLY.** You may read video files under `datasets/`. You may not
   write, move, rename, or delete anything there.
4. **Add `test/.venv/`, `test/.venv-intel/`, `test/results/`, and `test/_cache/` to `test/.gitignore`.**
   Do not modify the repo-root `.gitignore`.
5. Everything runs in **WSL**. State the WSL distro and kernel in the report.
6. **Time-box each track.** If a path is not working after the stated budget, record
   it as a negative finding and move on. Negative results are results.

---

## CONTEXT — WHY THIS EXISTS

Current pipeline decodes video on the fly with cv2. At `seq_len=4` it is tolerable;
at 8 it bottlenecks; the experimental plan requires **16 frames at stride 4**, which
spans ~61 source frames per clip. Decode cost scales with that span.

Three candidate directions, and I want all three measured rather than argued about:

- **CPU** — more workers, `cv2.setNumThreads(0)`, decord, multiprocessing over 20 cores
- **NVIDIA NVDEC** — dedicated decode silicon on the RTX 5080, idle during training
  because training saturates the SMs, not the decode engine
- **Intel iGPU** — QSV/VA-API decode, and `torch.xpu` compute

Known prior art in this repo: `books/research_notes.md` records that `ffmpeg_qsv` was
tested and rejected on this corpus (repeated decode failures, no decisive throughput
win). Your job is to re-measure, not to assume that verdict still holds — but flag any
result that contradicts it explicitly.

Known caveat to test, not assume: NVDEC has per-call overhead and does **not** always
beat software decode. PyTorch's own published benchmark shows software decoding winning
on 240x320 input and hardware pulling ahead at higher resolution. Our corpus is mostly
>=480p, so measure across the actual resolution distribution.

---

## PHASE 0 — ENVIRONMENT PROBE (write `test/results/env.json` first)

Capture and record, without failing on any missing item:

- WSL distro, kernel, `wsl --version` if reachable
- CPU model, physical/logical core count, RAM
- `nvidia-smi` full output: GPU name, driver, CUDA version, total/free VRAM
- NVDEC presence: `ffmpeg -decoders | grep -i nvidia` (expect `h264_cuvid`)
- NVDEC functional check:
  `ffmpeg -hwaccel cuda -hwaccel_output_format cuda -f lavfi -i testsrc2=duration=1 -f null -`
- `libnvrtc` presence
- Intel GPU: `lspci | grep -i vga`, `ls -l /dev/dxg`, `ls -l /dev/dri` (expect dri absent
  in WSL2 — record that as a finding, not an error)
- `vainfo` with and without `LIBVA_DRIVER_NAME=d3d12`
- `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
- `python -c "import torch; print(hasattr(torch,'xpu') and torch.xpu.is_available())"`
- ffmpeg version + configuration flags
- Installed versions of every package you end up using

**Corpus profile** — sample the real datasets and record the distribution of:
resolution, codec, container, fps, duration, total frame count, bitrate.
Benchmarks must be interpreted against this profile.

---

## SAMPLING & MEASUREMENT PROTOCOL (applies to every benchmark)

**Video sample set.** Build one fixed, stratified sample from the real corpus:
- >= 60 videos, drawn across `celeb-df-v2`, `faceforensics++`, `real-ai-videos`
- stratified across the resolution buckets found in the corpus profile
- write the exact file list to `test/results/sample_set.json` and reuse it for
  **every** benchmark, so all numbers are comparable
- include at least 3 videos known to appear in `train/video/bad_video_log.jsonl`,
  to see how each backend handles files that fail under cv2

**Page cache is the biggest confound.** Control it explicitly:
- report both **cold-cache** and **warm-cache** numbers
- for cold, either drop caches (`sync; echo 3 > /proc/sys/vm/drop_caches` if permitted)
  or use disjoint video subsets per repetition
- state which method you used; never silently report only warm numbers

**Timing discipline:**
- >= 3 warm-up iterations, discarded
- >= 10 timed iterations
- report **median and p95**, not mean alone
- `torch.cuda.synchronize()` before/after any GPU timing
- report throughput in **clips/sec** and **frames/sec**, plus per-clip latency

**Two load conditions, both required:**
1. **Idle** — decode with nothing else running
2. **Under concurrent GPU load** — decode while a dummy training loop saturates the SMs
   (large repeated matmul or a real ConvNeXt-B forward/backward at your target batch size)

Condition 2 is the realistic one. A decode path that wins when idle and collapses under
training load is useless to us. **Report both, and lead with (2) in the recommendation.**

---

## CORRECTNESS GATE — FRAME EQUIVALENCE (mandatory, gates everything)

For every alternative decode backend, compare against the **cv2 baseline** on the same
video and the same frame indices:

- exact frame-index alignment (off-by-one seek behaviour is a common and silent bug)
- per-pixel difference vs cv2: mean abs error, max abs error, PSNR
- colour space and channel order (BGR vs RGB), dtype, value range
- behaviour on the known-bad videos: does it raise, return garbage, or silently
  return zeros?

Classify each backend as:
- **EQUIVALENT** (max abs diff <= 2, no index drift)
- **NEAR-EQUIVALENT** (small colour-conversion delta; quantify it, state the risk)
- **DIVERGENT** (index drift or large pixel delta — disqualifying regardless of speed)

A DIVERGENT backend must be reported as unusable even if it is the fastest.

---

## TEST MATRIX

Every decode benchmark runs across: `seq_len in {4, 8, 16}` x `stride in {1, 4}`.
Note which combinations matter most: **(16, 4)** is the target, **(4, 1)** is the
current baseline.

### Track A — CPU baselines (budget: 2h)
- `A1` cv2 sequential, `cv2.setNumThreads` default
- `A2` cv2 with `cv2.setNumThreads(0)` inside workers
- `A3` cv2 + `DataLoader` worker sweep: `num_workers in {0,4,8,12,16,20}`,
  `persistent_workers`, `prefetch_factor in {2,4,8}`, `pin_memory`
- `A4` decord CPU
- `A5` PyAV / imageio-ffmpeg, if cheap to add
- `A6` multiprocessing pool over whole videos (the extraction-pass pattern, not the
  training pattern) — this is the one that matters for one-time preprocessing

### Track B — NVIDIA NVDEC (budget: 4h)
- `B1` **torchcodec** with `device="cuda"` — this is the primary candidate; PyTorch is
  consolidating video decode into TorchCodec and torchaudio's StreamReader is deprecated
- `B2` ffmpeg CLI with `-hwaccel cuda -hwaccel_output_format cuda`, piped
- `B3` NVIDIA DALI video reader, if installable without excessive pain
- `B4` decord with `ctx=decord.gpu(0)`, if it builds
- `B5` **NVDEC session/worker sweep** — decode contexts per process,
  `num_workers in {1,2,4,8}`. Find where contention starts. Report any session limit hit.
- `B6` **VRAM accounting** — peak VRAM used by decode surfaces alone, and decode+training
  concurrently at target batch size. Report headroom remaining.

### Track C — Intel iGPU (budget: 4h, hard stop)
Expect much of this to fail in WSL2. Failures are findings; document precisely.
- `C1` Is there an Intel GPU at all, and which one (Arc discrete / Arc integrated /
  Iris Xe / UHD)? This determines whether anything else is worth attempting.
- `C2` VA-API via the `d3d12` Mesa bridge: `LIBVA_DRIVER_NAME=d3d12 vainfo`.
  Record enumerated profiles. **Then actually attempt an ffmpeg VA-API decode** — the
  documented failure mode is that `vainfo` lists working profiles while ffmpeg still
  fails with the same parameters. Report both results separately.
- `C3` QSV decode attempt via ffmpeg
- `C4a` **Native `torch.xpu` first** — this is the baseline and the only long-term
  adoptable path, since XPU support is upstream in PyTorch. Check
  `torch.xpu.is_available()`, `torch.xpu.device_count()`, device properties. If present,
  benchmark a ConvNeXt-B forward pass vs the same on CUDA.
- `C4b` **`intel-extension-for-pytorch` (IPEX), in an ISOLATED venv** — `test/.venv-intel`,
  NOT the main `test/.venv`.

  **Why isolated:** IPEX xpu wheels pin specific torch builds from Intel's own index
  (`+cxx11.abi` variants via `https://pytorch-extension.intel.com/release-whl/...`).
  Installing them alongside your CUDA torch will break one or both. Never mix them in
  one environment.

  **Status to record in the report:** IPEX reached end of life at the end of March 2026.
  Intel stopped quarterly releases and no longer publishes official binary wheels; the
  project is maintained only to let dependants migrate off it. Pre-EOL wheels remain
  installable from the existing index.

  Test it anyway — measure the last available `+xpu` release against C4a on the same
  ConvNeXt-B benchmark. Record the exact version installed and whether it built/ran.

  **Adoptability rule:** if IPEX beats native `torch.xpu` on this hardware, report the
  delta but mark the result `[MEASURED, NOT ADOPTABLE]` — we will not take a hard
  dependency on an EOL package with no security patches for a project that runs into
  2027. It would only justify vendoring if the gap were very large AND native XPU were
  unusable. Say so plainly rather than recommending it.
  If IPEX fails to install or run, that is a clean negative finding — record the exact
  error, do not fight the build past the Track C budget.
- `C5` OpenVINO face detection throughput on Intel GPU vs CUDA, if C1 found usable hardware.
  OpenVINO is separately maintained and is NOT affected by the IPEX EOL, so a win here
  IS adoptable — note that distinction in the report.

### Track D — Face detection (budget: 3h)
This drives the one-time extraction pass over ~12k videos.
- `D1` candidate detectors on CUDA: compare at least two of
  RetinaFace / SCRFD / MTCNN / YuNet — throughput and VRAM
- `D2` **fused GPU pipeline**: NVDEC decode -> detection on SMs, never leaving VRAM.
  Compare against decode-to-CPU -> upload -> detect.
- `D3` end-to-end extrapolation: given measured rate, project wall-clock to process
  12,028 videos at 48 crops/video, and estimate output size on disk
- `D4` detection quality sanity check: face-found rate per detector on a sample.
  A detector that is 3x faster but finds 20% fewer faces is not faster.

### Track E — Training-side throughput (budget: 2h)
Independent of decode; measures the compute side.
- `E1` AMP (fp16 / bf16) vs fp32
- `E2` `channels_last` memory format
- `E3` `torch.compile` — measure compile time separately from steady-state
- `E4` batch-size sweep at `seq_len=16` until OOM; report max stable batch and
  throughput at each
- `E5` gradient checkpointing trade-off at seq=16, if VRAM-limited

---

## DELIVERABLES

```
test/
├── .gitignore
├── requirements.txt          # self-contained, pinned (main env, CUDA torch)
├── requirements-intel.txt    # IPEX track ONLY — installs into test/.venv-intel
├── README.md                 # how to set up and run everything
├── common/
│   ├── env_probe.py
│   ├── sampling.py           # builds/loads the fixed sample set
│   ├── timing.py             # warm-up, repeats, median/p95, sync
│   ├── load_generator.py     # concurrent-GPU-load harness
│   └── correctness.py        # frame-equivalence checks vs cv2
├── decode/                   # Tracks A, B, C decode benchmarks
├── compute/                  # Tracks C4/C5, E
├── facedet/                  # Track D
├── run_all.py                # runs everything, survives individual failures
└── results/
    ├── env.json
    ├── corpus_profile.json
    ├── sample_set.json
    ├── <benchmark>.json      # one per benchmark, machine-readable
    └── REPORT.md
```

Every benchmark writes a JSON with a consistent schema:
`{name, track, config, condition (idle|under_load), seq_len, stride, cache_state,
median_ms, p95_ms, clips_per_sec, frames_per_sec, peak_vram_mb, correctness_class,
n_videos, n_iterations, errors[], notes}`

`run_all.py` must **continue past failures** and record them, so one broken track does
not kill the run.

---

## `test/results/REPORT.md` — REQUIRED STRUCTURE

```
1. Verdict                  (<=250 words: what to adopt, what to drop, what needs more work)
2. Environment              (hardware, drivers, versions, WSL details)
3. Corpus Profile           (resolution/codec/fps distribution the numbers apply to)
4. Correctness Results      (equivalence class per backend — BEFORE any speed table)
5. Decode Throughput        (matrix: backend x seq_len x stride x load-condition,
                             cold and warm cache)
6. Dataloader Tuning        (worker/prefetch sweep, the recommended config)
7. NVDEC Findings           (incl. session limits, VRAM cost, behaviour under load)
8. Intel Findings           (native torch.xpu vs IPEX vs OpenVINO; what failed, exact
                             error output; adoptability flag on every IPEX result)
9. Face Detection           (throughput, quality, projected extraction wall-clock)
10. Training Throughput      (AMP / channels_last / compile / batch sweep)
11. Recommendations          (ranked, each with the measurement that justifies it)
12. What I Could Not Test    (and what would be needed to test it)
```

Rules for the report:
- **Correctness section comes before every speed table.** A DIVERGENT backend is
  reported as unusable regardless of throughput.
- Lead recommendations with **under-load** numbers, not idle.
- Give speedups as multiples against the A1 cv2 baseline at the same seq/stride/cache state.
- Where a result contradicts `books/research_notes.md` (e.g. QSV), say so explicitly
  and show the evidence.
- No recommendation without a cited measurement.

---

## RULES

- Nothing outside `test/` changes. If you believe something outside must change, stop
  and tell me instead.
- Datasets are read-only.
- Respect the per-track time budgets. Record what you skipped and why.
- If a dependency will not install in WSL, record the exact error and move on. Do not
  spend an hour fighting a build.
- Never report a synthetic or stubbed number as a measurement. If you could not run
  something, say `[NOT MEASURED]`.
- Report negative results with the same care as positive ones — "Intel QSV does not work
  in WSL2, here is the exact failure" is a genuinely useful outcome.
- Do not adopt, integrate, or refactor anything into the main pipeline. Measure and
  recommend only.

Start with Phase 0 and the corpus profile, show me `env.json` and the sample set before
running the full matrix.
