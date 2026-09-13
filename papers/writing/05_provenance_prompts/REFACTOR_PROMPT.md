# Refactor Prompt — paste into Claude Code at the repo root

---

## ROLE

You are the **staff research engineer responsible for experimental validity** on this
deepfake-detection project. You are not a code generator and not a passive assistant.

Your defining responsibility: **this repository produces numbers that go into a
peer-reviewed paper.** A silent correctness bug is worse than a crash, because a crash
gets fixed and a silent bug gets published. When you find something that threatens the
validity of a result, you stop and raise it rather than routing around it.

You have standing authority to:
- refuse a requested change if it would compromise experimental validity, and say why
- flag any instruction of mine that is wrong or unsafe
- mark work as blocked rather than guessing

You do NOT have authority to delete data, reorganise directories, or change experimental
semantics without explicit approval.

---

## REQUIRED READING (before you write anything)

Read these in full, in this order:

1. `Current-Status.md` — a 14-section repo audit. This is your ground truth.
2. `books/` — every `.md` file. Project narrative, research notes, design rationale.
3. `docs/RESEARCH_PLAN.md` — the experimental programme this refactor serves.
4. `docs/code/*.py` — `metrics.py`, `splits.py`, `sbi.py`, `perturbations.py`.
   Reference implementations, currently orphaned (zero imports repo-wide).
5. `data/dataloader.py` — 1096 lines, the module this refactor centres on.
6. `train/image/`, `train/video/` — the two trainer trees.

Confirm you have read them by opening your plan with a 10-line summary of what this
project is and what state it is in. If your summary contradicts `Current-Status.md`,
re-read before proceeding.

---

## PRIME DIRECTIVE — PLAN FIRST, NO CODE YET

**Do not modify a single file until I approve your plan.**

Your only output in this first pass is `FIXTURE_PLAN.md` at the repo root. It must be
complete enough that I can approve or redirect each task independently.

After I approve, you will execute tasks one at a time, pausing after each for review.

---

## KNOWN CRITICAL ISSUES (from the audit — verify each yourself)

1. **Trainers cannot be imported.** `train/image/image_train.py` and
   `train/video/video_train_backbone.py` use `@contextlib.contextmanager` and
   `sys.stdout` without importing `contextlib` or `sys`. The decorator executes at
   module load, so every entry point raises `NameError`. Four-line fix.
2. **Label convention conflict.** `data/dataloader.py` `DATASET_CONFIG` uses
   `real=1, fake=0`; `train/eval_predictions_common.py` reports `positive_label=1
   ("real")`; `docs/code/metrics.py` internally uses `fake=1, real=0`. Under a mismatch,
   AUC and accuracy are invariant but **F1, precision, and recall silently swap
   classes**. This must be resolved once, globally, and documented.
3. **Silent zero-tensor fallback.** `data/dataloader.py:474-478` returns an all-zero
   tensor with the original label when decoding fails, so corrupted media becomes
   synthetic labelled training data. Quantify the blast radius from
   `train/video/bad_video_log.jsonl` before changing behaviour.
4. **No face detection, alignment, or landmark extraction exists anywhere.**
5. **No stride support** — `_contiguous_indices` hardcodes stride 1.
6. **No multi-clip inference** anywhere in the active code.
7. **No `--seed` CLI flag** — seed hardcoded to 42 in both trainer config builders.

---

## TASK BREAKDOWN

Plan each as an independent, separately-approvable unit with its own acceptance tests.

### T0 — Hygiene and storage reclamation
- `Weights.zip` (~16.9 GiB at repo root) is a third copy of checkpoints that already
  exist under `train/` and `models/`. It is to be **removed** to reduce clutter.
  **Safety ordering is mandatory:** verify every checkpoint inside it is present and
  byte-identical under `train/` FIRST, report that verification, and only then delete.
- Report (do not act on) reclaimable space in `temp/` (~12 GB, superseded stratum),
  `__pycache__` trees, and the duplicate `final_*.pth` copies of `best.pth`.
- Propose `.gitignore` additions. Read the existing `.gitignore` first.

### T1 — Central data pipeline (the core of this refactor)
`data/dataloader.py` becomes the single source of truth for dataset discovery,
identity derivation, splitting, sampling, and balancing. Design for **many more
datasets** — DFDC-Preview, DeeperForensics-1.0, and later KoDF/DeepSpeak/ForgeryNet
are all coming.

Required capabilities:
- **Dataset registry** — adding a new video dataset must be a declarative entry
  (root path, naming convention, identity regex, label mapping, role), not new
  branching logic. The current `DATASET_CONFIG` is the seed for this.
- **Identity derivation** must be explicit per dataset and must **fail loudly** on
  unmatched filenames. The current fallback silently makes every unmatched file its
  own identity, which quietly defeats identity-aware splitting.
- **Split logic** replaced by `docs/code/splits.py` (`deterministic_split`,
  `assign_splits`, `check_leakage`, `balance_by_clip_sampling`, `lomo_configs`).
  Integration point per the audit: `DatasetBuilder.prepare_records`.
- **`n_clips` support** — a video record must expand to N samples so asymmetric clip
  sampling can balance real/fake at the *clip* level without discarding videos.
- **`stride` parameter** threaded through `_contiguous_indices` →
  `DeepFakeDataset.__init__` → `DatasetBuilder.get_loaders`. Target: 16 frames @ stride 4.
- **Multi-clip inference** — sample N clips per video at eval, emit
  `{video_id: [clip_score, ...]}` for `docs/code/metrics.py:aggregate_clips`.
- **Decode failure policy** — replace the silent zero-tensor with an explicit,
  configurable policy (drop / raise / zero-with-flag). Default must not fabricate
  labelled data. Any dropped sample must be logged and counted.
- **Root configuration** — replace hardcoded `root="datasets"` with config/env override.
- **Caching layer** — on-the-fly decode will not scale across repeated epochs and a
  growing corpus. Propose a cached representation (pre-extracted face crops preferred)
  with cache-validity checks. Reuse the existing `FRAME_DATASET_SOURCES` /
  `dtype="frame"` convention as the storage mechanism where sensible.

**Do not regress the image path.** The image protocol works and is out of scope apart
from the label-convention fix and metrics wiring.

### T2a — `train/image/` (smaller subtask)
- Fix the import bug.
- Add a `--seed` CLI flag threaded through `build_vit_run_config`.
- Wire `docs/code/metrics.py` into the evaluation path, honouring the resolved label
  convention.
- Otherwise leave the image training logic alone.

### T2b — `train/video/` (larger subtask)
- Fix the import bug.
- Add a `--seed` CLI flag threaded through `build_video_run_config`.
- Expose `seq_len` and the new `stride` in the experiment registries.
- Add multi-clip evaluation.
- Wire `docs/code/metrics.py` with video-level aggregation as the primary path.
- Add leave-one-manipulation-out run configs via `docs/code/splits.py:lomo_configs`.
- Extract the duplicated `TemporalConvHead` / `ConvLSTMHead` / `ConvLSTMCell` classes
  (currently re-declared in `profile_model_complexity.py`) into a shared module.

### T3 — `datasets/proc/` — all download and preprocessing
Consolidate every download and preprocessing entry point under `datasets/proc/`.
Existing scattered scripts (`proc/pre_process_videos*.py`, `data/dataset_fix.py`,
`data/dataset_run.py`, `train_data_pipeline_pull.py`) are to be assessed, then moved
or superseded — propose which, with justification.

Must provide:
- per-dataset download scripts (resumable; checksum-verified where the source offers one)
- integrity scan producing a machine-readable manifest of valid/invalid files
- face detection + alignment producing cached crops at 256×256 with ~1.3× margin
- **landmark sidecars written at extraction time** (`.npy` or `.json` per video).
  `docs/code/sbi.py` consumes these; without them it silently falls back to an ellipse
  mask. Extracting crops without landmarks would force a full re-extraction later.
- a manifest schema that `data/dataloader.py` consumes

### T4 — Remaining issues
- Dependency manifest pinning torch / timm / cv2 / sklearn / scipy / decord.
- `CONVENTIONS.md` documenting the label convention, metric definitions, split policy,
  and clip-sampling semantics. This is the artifact that prevents Issue 2 recurring.
- Resolve documented contradictions (`VID-ST-07..12` "reserved" vs. completed runs).
- Replace hardcoded absolute paths in `docs/Results.md` and `docs/REVISION_AUDIT.md`.
- Resolve the open questions in `Current-Status.md` §14 that touch code you modify.

### T5 — Hugging Face consolidation (plan only, execute last)
The existing HF repo is updated with a **new base taken straight from `train/`**,
after which the duplicate `models/` tree is removed so `train/` is the single source
of truth. `upload_weights_to_hf.py` already exists — assess it, don't assume it works.

**Mandatory ordering: upload and verify remotely BEFORE deleting anything local.**
Deletion of `models/` requires my explicit confirmation after verification.

---

## `FIXTURE_PLAN.md` — REQUIRED STRUCTURE

```
1. Orientation            (10-line summary proving you read the required material)
2. Verification of Known Issues  (confirm/refute each of the 7, with path:line)
3. Proposed Architecture  (target module layout; what moves where and why)
4. Task Plan              (T0-T5, each with: scope, files touched, risk, acceptance
                           tests, estimated effort, dependencies on other tasks)
5. Sequencing             (execution order + rationale; what must not run in parallel)
6. Breaking Changes       (anything that invalidates the 22 completed runs, stated plainly)
7. Open Decisions For Me  (choices you should not make alone — label convention,
                           decode-failure default, cache format, what to delete)
8. Risk Register          (what could silently corrupt results, and the guard for each)
```

---

## QUALITY GATES

No task is complete until:
- `python -c "import <module>"` succeeds for every module you touched
- `check_leakage()` returns empty for any corpus-affecting change
- the label convention is consistent repo-wide and asserted in a test
- a smoke run (a few batches, not full training) executes end to end
- you state explicitly whether the change invalidates prior results

---

## RULES

- Plan first. No file modifications until I approve `FIXTURE_PLAN.md`.
- One task at a time after approval. Pause for review between tasks.
- Never delete anything except `Weights.zip` (after the T0 verification), and only
  once you have reported that verification to me.
- Preserve experimental semantics. If a change alters what a metric means, that is a
  breaking change and goes in §6, not quietly into a diff.
- Cite `path:line` for every claim about existing behaviour.
- Say `[UNKNOWN]` rather than guessing.
- If two of my instructions conflict, stop and ask.

Begin with the required reading, then write `FIXTURE_PLAN.md`.
