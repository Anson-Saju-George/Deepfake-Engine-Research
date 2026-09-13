# Master Benchmark Re-run (REPORT v2) — paste into Claude Code, run in WSL

## ROLE
Re-run the FULL Track A–E hardware benchmark in the real training environment `/opt/ml`
and produce `test/results/REPORT_v2.md`. Two things changed since REPORT.md, and both
invalidate parts of it:

1. **cuDNN.** REPORT.md ran in a disposable venv (`torch 2.13.0+cu132`, deleted) that hit
   `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` and forced `cudnn.enabled = False`. Every
   GPU-compute number there is a FLOOR. `test/results/CUDA_VERIFY.md` certified `/opt/ml`
   CLEAN with cuDNN enabled (5/5 probes incl. ConvNeXt fwd+bwd).
2. **Intel GPU compute now works.** REPORT.md §8 claims *every* Intel path fails because
   `/dev/dri` is missing. That is now **factually wrong for compute**. After adding
   Intel's own apt repo (`repositories.intel.com/gpu/ubuntu noble client`) and installing
   `intel-opencl-icd intel-level-zero-gpu libze1 clinfo`, `clinfo` enumerates a real
   device: **Intel(R) Graphics [0x7d67], type GPU, 64 max compute units, 33.54 GB shared
   memory, Available: Yes**. The VA-API/QSV *video decode* finding still stands — that's a
   separate subsystem that genuinely needs `/dev/dri`. Track C must be rewritten, not
   re-run as-is.

Also: `sudo` is now available, so `vainfo` (previously `[NOT MEASURED]`) can be installed.

## CONSTRAINTS
1. **RUN IN `/opt/ml`.** Confirm interpreter is `/opt/ml/bin/python` before any benchmark.
   If `torch.__version__` contains `+cu132`, STOP — wrong env.
2. **NO PACKAGE MUTATIONS DURING THE RUN.** This is a *methodology* rule, not an
   environment policy: changing versions mid-sweep makes tracks non-comparable. Do not
   pip install/upgrade/uninstall, do not create venvs, do not source
   `test/requirements*.txt` at any point between pre-flight and the final report.
   EXCEPTION: `sudo apt install -y vainfo` is permitted **during pre-flight only**, before
   any measurement starts. Record it.
3. **cuDNN MUST BE ENABLED** for every GPU track. Assert
   `torch.backends.cudnn.enabled == True` and log it per track. Strip any
   `cudnn.enabled = False` line for this run and note where it was. A cuDNN-off number is
   a failed measurement, not a result.
4. **DATASETS READ-ONLY.** No writes/moves/renames/deletes under `datasets/`.
5. **WRITE ONLY UNDER `test/`.** Nothing outside `test/` changes.
6. Code edits limited to: removing `cudnn.enabled=False`, removing package-install/venv
   calls, and the new Track C tests below. No other refactors. If a script needs more,
   record `[NOT RE-RUN]` with the reason.

## PRE-FLIGHT → `test/results/rerun_env.json`
```
which python                      # must be /opt/ml/bin/python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled, torch.cuda.get_device_name(0))"
python -c "import cv2; print(cv2.__version__, cv2.ocl.haveOpenCL(), cv2.ocl.useOpenCL())"
clinfo --list
sudo apt install -y vainfo && vainfo   # permitted here only
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
pip freeze > test/results/rerun_pip_freeze.txt
```
Grep every A–E script for `cudnn.enabled`, `pip install`, `venv` — list what you found and
neutralised BEFORE running anything.

**IMPORTANT — new confound:** installing `intel-opencl-icd` may make OpenCV auto-detect an
OpenCL device where it previously found none, which can silently change Track A cv2
numbers vs REPORT.md. Call `cv2.ocl.setUseOpenCL(False)` in every Track A/B decode
benchmark for comparability with the prior run, and log that you did. Then, separately,
measure the OpenCL-enabled variant as a NEW datapoint (see C7).

## TRACKS

| Track | Re-run? | Why |
|---|---|---|
| A CPU decode | Yes | completeness; expect to match REPORT.md (with OpenCL off) |
| B NVDEC decode | Yes | confirm "slower than pyav" verdict holds with cuDNN on |
| C Intel | **REWRITE** | compute path now works — see below |
| D face detection | Yes — priority | MTCNN ran cuDNN-off before |
| E training throughput | Yes — TOP priority | the main reason for this re-run |

If time runs short: **E and D must complete.** A/B may be truncated with a note. C is
bounded at 3h, hard stop.

### Track E (top priority)
- fp32 / channels_last / AMP fp16 / AMP bf16 / AMP bf16+channels_last.
- **Re-test `torch.compile` with cuDNN ON** — prior 10.6x slowdown was hypothesised to be
  the cuDNN-off fallback. Report compile time separately from steady state.
- **Batch sweep past 4** — prior ceiling came from cuDNN-off kernels near OOM. Find the
  real max stable batch at seq_len=16, throughput at each.
- bf16 vs fp16: if within noise, **recommend bf16** (loss-scaling-free, NaN-safe on dark/
  compressed frames). State this explicitly.

### Track D
- MTCNN-CUDA with cuDNN on vs YuNet-CPU. Include face-found rate, not just speed.
- Note that extraction is a one-time ~1h pass, so simplicity may still favour YuNet even
  if MTCNN closes the gap. Give the reasoning.

### Track C — REWRITTEN (budget 3h, hard stop)
Correct the record first, then test what's genuinely new.

- **C1 Correction.** State plainly that REPORT.md §8's "every Intel path fails" is wrong
  for compute. Give the exact `clinfo` device block as evidence. Keep the VA-API/QSV
  decode finding — re-confirm it quickly with `vainfo` (now installable) so the split
  conclusion is evidence-backed: *decode dead (`/dev/dri`), compute alive (dxgkrnl)*.
- **C2/C3.** Quick re-confirm VA-API and QSV still fail. Do not fight them.
- **C5 OpenVINO GPU — RETEST, highest value in this track.** Previously `available_devices`
  showed CPU only, measured BEFORE this driver stack existed. Re-check now. OpenVINO is
  NOT EOL-affected, so a win here IS adoptable. If a GPU device appears, benchmark a face
  detection model on it vs CPU.
- **C6 NEW — OpenCV DNN with OpenCL target.** This is the one genuinely adoptable path:
  run YuNet via `cv2.FaceDetectorYN` with `DNN_TARGET_OPENCL` / `DNN_TARGET_OPENCL_FP16`
  and compare against the CPU target from Track D. No PyTorch involvement, no wheel
  conflicts, works with the stack you just installed.
- **C7 NEW — raw compute throughput.** Measure the iGPU's actual FP32 capability
  (e.g. a large GEMM via pyopencl **if already installed** — if not, do NOT install it;
  record `[NOT MEASURED]`). Compare against the same GEMM on the RTX 5080 to get a real
  ratio instead of an estimate. 64 compute units is a small part; quantify it.
- **C4a native `torch.xpu`.** Confirm `torch.xpu.is_available()` in `/opt/ml` (expected
  False — this is a CUDA-only wheel). Do NOT install an XPU torch build: it is mutually
  exclusive with `torch+cu129` and would break the training env. Record as
  `[NOT TESTABLE IN THIS VENV]` with that reason.
- **C4b IPEX.** Bounded retry only if trivial. It is EOL (March 2026) and the prior
  failure was a wheel-extraction bug (`libtorch_global_deps.so` missing), not a driver
  problem, so new drivers are unlikely to fix it. Mark any result
  `[MEASURED, NOT ADOPTABLE]`. Do not exceed 30 minutes here.

## REPORT → `test/results/REPORT_v2.md`
Mirror REPORT.md's structure for comparability, plus:
- **§0 Corrections vs REPORT.md** — at the top. Two subsections: (a) cuDNN-off numbers now
  re-measured, old vs new side by side with deltas; (b) the Intel §8 correction, stated
  plainly as an error in the prior report with the evidence that overturns it.
- Per GPU track, state `cudnn.enabled` was True (cite the logged value).
- State whether `cv2.ocl.useOpenCL()` was False for A/B (comparability) and note C6
  separately.
- **Correctness results before any speed table** (unchanged rule).
- Lead recommendations with **under-load** numbers.
- Mark anything skipped `[NOT RE-RUN]` / `[NOT MEASURED]` with the reason.
- **Final one-line verdict** on each: decode backend, worker count, AMP dtype,
  torch.compile, face detector, max batch size, and Intel GPU (adoptable or not, and for
  what specifically).

## RULES
- `/opt/ml` only. No package mutations after pre-flight. cuDNN on. Datasets read-only.
- Log continuously to a file; timestamped progress line per track so the morning log reads
  cleanly.
- Never report a cuDNN-off, OpenCL-confounded, or stubbed number as a real measurement.
- If forced to choose, **finish E cleanly** over finishing everything.
- Own any place the prior report was wrong. Corrections are the point of v2.

Start with pre-flight, confirm `/opt/ml` + cuDNN True + cv2 OpenCL state, list what you
neutralised, then run A→E.
