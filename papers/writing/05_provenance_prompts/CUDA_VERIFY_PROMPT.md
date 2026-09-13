# CUDA / cuDNN Verification Prompt — paste into Claude Code, run inside WSL

---

## ROLE

You are diagnosing whether the **real training environment** on this machine can run
GPU convolutions with cuDNN enabled. This is a **read-only verification task**, not a
setup task.

A benchmark run earlier (in a now-deleted throwaway venv, `test/.venv`, using an
experimental `torch 2.13.0+cu132` build) hit `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`
on ordinary `conv2d` and worked around it with `torch.backends.cudnn.enabled = False`.
The question now: **does that mismatch also affect the real training environment, or was
it isolated to that disposable venv?**

Preliminary evidence suggests the real env is healthy (`torch 2.11.0+cu129`, cuDNN
92500, `is_available()==True`), but `is_available()` only proves the GPU is visible — it
does NOT prove `conv2d` runs with cuDNN on. That is the gap to close.

---

## ABSOLUTE CONSTRAINTS

1. **DO NOT install, upgrade, downgrade, or uninstall anything.** Not pip, not apt, not
   conda, not a driver, not a toolkit, not cuDNN. Zero package mutations.
2. **DO NOT set or export environment variables** that persist. Read-only probes only.
3. **DO NOT modify any file** except writing the final report to
   `test/results/CUDA_VERIFY.md`.
4. If a check fails, **report the failure with its full traceback** — do not attempt to
   fix it. Diagnosis is the deliverable, not remediation.
5. Run everything **inside WSL**, in the **real training environment** — the same
   Python/venv the actual `train/` code uses. NOT `test/.venv` (deleted) and NOT
   `test/.venv-intel`. If you are unsure which interpreter that is, determine it first
   (see Phase 0) and state it explicitly before running anything.

---

## PHASE 0 — IDENTIFY THE RIGHT INTERPRETER (do this first, report before proceeding)

The failure was environment-specific, so running the wrong interpreter answers the wrong
question. Establish and report:

- `which python` and `python --version`
- Is there an active venv/conda env? Which one? (`echo $VIRTUAL_ENV`, `echo $CONDA_PREFIX`)
- How does the real training code get invoked — is there a documented venv in the repo
  README, a `requirements.txt`, an activate script? Find the interpreter that
  `python -m train.video.run_video` would actually use.
- `pip show torch` — location, version. Confirm it is `2.11.0+cu129` (or report what it
  actually is).

**Stop and report which interpreter you selected and why before running Phase 1.**
If there are multiple candidate environments, list them and probe the one the training
entry points use.

---

## PHASE 1 — STATIC ENVIRONMENT PROBE (no GPU ops yet)

Collect and record, each labeled clearly:

```python
import torch
torch.__version__
torch.version.cuda                       # runtime CUDA the wheel was built against
torch.backends.cudnn.version()           # bundled cuDNN
torch.backends.cudnn.enabled             # should be True
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)      # expect (12, 0) for RTX 5080 Blackwell/sm_120
```

Also capture (shell, read-only):
- `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv`
- `nvidia-smi | head -5` (driver + CUDA UMD version line)
- `python -c "import torch, os; print(os.path.dirname(torch.__file__))"` then list the
  bundled `nvidia/cudnn` wheel version under site-packages if present
  (`pip show nvidia-cudnn-cu12` — report version, do not change it)

Record whether `torch.cuda.get_device_capability` returns `sm_120`. If the wheel's
supported archs do not include sm_120, that is the actual root cause of any failure and
must be called out.

---

## PHASE 2 — THE DECISIVE TEST (GPU conv2d, cuDNN ENABLED)

This is the whole point. Run with cuDNN **enabled** (do not disable it):

```python
import torch, traceback
torch.backends.cudnn.enabled = True      # explicit; this is what we're testing

def probe(name, fn):
    try:
        fn(); torch.cuda.synchronize()
        print(f"PASS  {name}")
    except Exception:
        print(f"FAIL  {name}")
        traceback.print_exc()

x = torch.randn(8, 3, 224, 224, device="cuda")
w = torch.randn(64, 3, 7, 7, device="cuda")

# 1. plain matmul (does NOT use cudnn — isolates cudnn from general CUDA health)
probe("cuda matmul (no cudnn)", lambda: torch.randn(1024,1024,device="cuda") @ torch.randn(1024,1024,device="cuda"))

# 2. the exact op that crashed in the benchmark venv
probe("conv2d fp32 cudnn-enabled", lambda: torch.nn.functional.conv2d(x, w))

# 3. bf16 autocast conv2d — the path chosen for real training
def bf16_conv():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.nn.functional.conv2d(x, w)
probe("conv2d bf16 autocast", bf16_conv)

# 4. a real backbone forward — convnext_base, the actual model family
def backbone_fwd():
    import torchvision
    m = torchvision.models.convnext_base(weights=None).cuda().eval()
    with torch.no_grad():
        m(torch.randn(2,3,224,224,device="cuda"))
probe("convnext_base forward", backbone_fwd)

# 5. a real backward pass (training path, not just inference)
def backbone_bwd():
    import torchvision
    m = torchvision.models.convnext_base(weights=None).cuda().train()
    out = m(torch.randn(2,3,224,224,device="cuda"))
    out.sum().backward()
probe("convnext_base forward+backward", backbone_bwd)

print("cudnn.enabled at end:", torch.backends.cudnn.enabled)
print("cudnn.version:", torch.backends.cudnn.version())
```

Interpretation rules for the report:
- **matmul PASS + conv2d FAIL** → cuDNN-specific problem (the mismatch is real here).
- **matmul FAIL too** → broader CUDA/driver/arch problem, not just cuDNN.
- **all PASS** → the real training env is clean; the earlier mismatch was isolated to
  the deleted benchmark venv, and every cuDNN-disabled compute number in the prior
  report was a floor.

---

## PHASE 3 — VERDICT (`test/results/CUDA_VERIFY.md`)

```
1. Interpreter Used        (path, version, how selected, why it's the training env)
2. Static Probe            (torch/cuda/cudnn versions, device capability, driver)
3. conv2d Test Results     (PASS/FAIL table for all 5 probes, with any traceback)
4. Verdict                 (one of:
                            A. CLEAN — conv2d works with cudnn enabled, no action needed
                            B. CUDNN MISMATCH — conv2d fails, matmul works; state exact
                               error and the MINIMAL suggested fix (do not apply it)
                            C. BROADER CUDA ISSUE — matmul also fails; describe)
5. Implication for prior report
                           (if CLEAN: the §10 AMP/batch/compile numbers were measured
                            with cudnn disabled and are floors — real training will be
                            faster; torch.compile should be re-tested)
6. Recommended Action      (if CLEAN: none, freeze the env. if not: the smallest
                            possible change, proposed for MY approval — never applied
                            by you.)
```

---

## RULES

- Read-only. No installs, no upgrades, no env mutations, no persistent exports.
- If any probe fails, capture the FULL traceback verbatim — that error text is the
  diagnostic value.
- Do not "fix" anything. Propose the minimal fix in the report and stop.
- If you cannot find the real training interpreter with confidence, say so and list
  candidates rather than guessing.
- State plainly which of the three verdicts (A/B/C) the evidence supports.

Run Phase 0, report the interpreter choice, then proceed through Phases 1-3.
