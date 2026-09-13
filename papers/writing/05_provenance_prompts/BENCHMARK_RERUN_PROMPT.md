# Benchmark Re-run (REPORT 2.0) — paste into Claude Code, run inside WSL

## ROLE
Re-run the FULL Track A–E hardware benchmark (`test/`) in the real, certified-clean
training environment `/opt/ml` to produce `test/results/REPORT_v2.md`.

The original REPORT.md ran in a disposable venv (`test/.venv`, `torch 2.13.0+cu132`, now
deleted) that hit `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` and forced
`torch.backends.cudnn.enabled = False`. Every GPU-compute number there is a FLOOR
measured with cuDNN OFF. `test/results/CUDA_VERIFY.md` confirmed `/opt/ml` is CLEAN:
conv2d, bf16 autocast, and full ConvNeXt-Base forward+backward pass with cuDNN ENABLED.
Job: re-run everything in `/opt/ml` with cuDNN ON, report real numbers.

## ABSOLUTE CONSTRAINTS
1. RUN IN `/opt/ml` ONLY. Confirm interpreter is `/opt/ml/bin/python` before any
   benchmark. Do NOT create/activate/install into any other venv. Do NOT recreate
   `test/.venv` or `test/.venv-intel`.
2. ZERO PACKAGE MUTATIONS. No pip install/uninstall/upgrade, no `python -m venv`, no
   sourcing `test/requirements*.txt`. `/opt/ml` is frozen/production. If any script tries
   to install or venv-create on startup, patch that line out or skip it and record it.
   Mutating `/opt/ml` is the single worst possible outcome of this run.
3. cuDNN MUST BE ENABLED. Before every GPU track assert
   `torch.backends.cudnn.enabled == True` and print it to the log. If any script sets it
   False, remove that line for this run and note it. A cuDNN-off GPU number is a failed
   measurement — flag it, don't report it as real.
4. DATASETS READ-ONLY. No writes/moves/renames/deletes under `datasets/`.
5. WRITE ONLY UNDER `test/` (`test/results/`, `test/_cache/`, logs). Nothing outside
   `test/` changes.
6. CODE CHANGES limited to removing `cudnn.enabled=False` lines and any
   package-install/venv-create calls. No other refactors. If a script needs more than
   that, record `[NOT RE-RUN]` with the reason instead of rewriting it.

## PRE-FLIGHT (log to `test/results/rerun_env.json`, verify before Track A)
```
which python                      # must be /opt/ml/bin/python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled, torch.cuda.get_device_name(0))"
```
Expected: `2.11.0+cu129 12.9 92500 True NVIDIA GeForce RTX 5080 Laptop GPU`.
If `torch.__version__` contains `+cu132`, STOP — wrong env.
Also record driver version, free VRAM at start, timestamp. Grep every A–E script for
`cudnn.enabled`, `pip install`, `venv` — list what you found and what you neutralised
BEFORE running anything.

## WHAT TO RE-RUN, AND WHY
Re-run all of A–E for a clean, footnote-free report. Weight effort where cuDNN changes
the answer:

| Track | cuDNN-sensitive | Priority |
|---|---|---|
| A CPU decode | No | completeness; should match REPORT.md |
| B NVDEC decode | Partially | confirm "slower than pyav" holds |
| C Intel iGPU | No (`/dev/dri` missing) | quick confirm it still fails; don't fight it |
| D face detection (MTCNN CUDA) | YES | priority — ran cuDNN-off before |
| E training throughput | YES — the whole point | TOP priority |

If time runs short, E and D MUST complete; A/B/C can be truncated with a note. Respect
original per-track budgets; if a track hangs, kill it, record `[NOT RE-RUN]`, move on.
Use the harness's existing timeout/hang protection.

### Track E specifics (the reason this re-run exists)
- Re-measure: fp32, channels_last, AMP fp16, AMP bf16, AMP bf16+channels_last.
- Re-test `torch.compile` with cuDNN ON — the prior 10.6x slowdown was likely the
  cuDNN-off fallback; it may flip to a win. Report compile time separately from
  steady-state.
- Batch-size sweep PAST 4 — prior ceiling of 4 came from cuDNN-off fallback kernels near
  OOM at 8. With cuDNN on, find the real max stable batch at seq_len=16 and report
  throughput at each.
- bf16 vs fp16: if within noise, RECOMMEND bf16 (loss-scaling-free, NaN-safe on
  dark/compressed frames). State this.

### Track D specifics
- Re-measure MTCNN-CUDA with cuDNN on vs YuNet-CPU.
- Even if MTCNN closes the gap, extraction is a one-time ~1h pass, so the recommendation
  may still favour YuNet on simplicity — give the reasoning, don't just pick the faster
  number.

## REPORT — `test/results/REPORT_v2.md`
Mirror REPORT.md structure for direct comparability, plus:
- Top section "Changes vs REPORT.md": every GPU number that moved, old (cuDNN-off) vs new
  (cuDNN-on) side by side, with delta.
- Per GPU track, state `cudnn.enabled` was True (cite the logged value).
- Correctness results BEFORE any speed table (unchanged rule).
- Lead recommendations with under-load numbers.
- Mark anything not re-run `[NOT RE-RUN]` with reason.
- One-line final verdict on each: decode backend, worker count, AMP dtype,
  torch.compile, face detector, max batch size.

## RULES
- `/opt/ml` only. No package mutations. cuDNN on. Datasets read-only. Writes under
  `test/` only.
- Log continuously to a file so an overnight hang leaves a diagnosable trail.
- Never report a cuDNN-off or stubbed number as real.
- If forced to choose between finishing E cleanly and finishing everything, FINISH E.
- Print a timestamped progress line per track for a readable morning log.

Start with pre-flight, confirm `/opt/ml` + cuDNN True, list what you neutralised, then
run A→E.
