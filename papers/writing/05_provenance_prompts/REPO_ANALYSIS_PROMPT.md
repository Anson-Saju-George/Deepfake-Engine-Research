# Repo Analysis Prompt — paste into Claude Code at the repo root

---

## ROLE

You are a senior research-engineering auditor. Your job is to produce a complete,
evidence-based inventory of this deepfake-detection research repository and write it
to `Current-Status.md` at the repo root.

This repo has been used for **two separate ground-up development cycles**, so expect
duplicated implementations, superseded modules, orphaned experiment code, and stale
artifacts. Surfacing those is a primary goal, not a side note.

The output document will be used as the **single source of truth** by another engineer
planning a rewrite. Accuracy matters more than completeness; unverified claims are
worse than admitted gaps.

---

## HARD CONSTRAINTS

1. **READ-ONLY.** Do not modify, move, delete, refactor, or reformat any existing file.
   The ONLY file you create is `Current-Status.md`.
2. **Do not run training, inference, or any long-running script.** You may run read-only
   shell commands (`tree`, `wc -l`, `ls`, `find`, `du`, `git log`, `grep`) and may import
   nothing.
3. **Do not guess.** Every factual claim must be traceable to a file, and where useful,
   a `path:line` reference. If you infer something, mark it explicitly as `[INFERRED]`.
   If you cannot determine something, write `[UNKNOWN]` and say what would resolve it.
4. **Do not summarize away specifics.** Exact function signatures, config keys, tensor
   shapes, and file counts are the point.

---

## PHASE 1 — DISCOVERY

Enumerate before analyzing.

1. Full directory tree (exclude `__pycache__`, `.git`, `node_modules`, and any
   directory over 500 files — note those separately with counts and total size).
2. Every `.py` file: path, line count, last-modified date.
3. Every `.md` file: path, line count, last-modified date.
4. Every config file: `.yaml`, `.yml`, `.json`, `.toml`, `.ini`, `.cfg`.
5. Every checkpoint/weight file: `.pt`, `.pth`, `.ckpt`, `.safetensors` — path and size.
6. Every results artifact: `.csv`, `.json` summaries, log files, plots.
7. Dataset directories: path, immediate subdirectory names, file counts per leaf,
   total size. **Do not enumerate individual media files.**
8. `git log --oneline -50` and current branch, if a git repo.

---

## PHASE 2 — READ EVERYTHING

Read **all** `.py` files and **all** `.md` files in full. For large files (>800 lines),
read fully but summarize structurally.

Prioritize in this order: training entry points → dataset/dataloader code → model
definitions → evaluation/metrics → utilities → plotting → docs.

---

## PHASE 3 — ANALYSIS DIMENSIONS

For each dimension, record what the code *actually does*, not what filenames suggest.

### A. Data pipeline
- Where dataset roots are configured (hardcoded paths? env vars? config files?)
- Dataset/Dataloader class names and **exact `__init__` and `__getitem__` signatures**
- What `__getitem__` returns: types, shapes, dtype, value range, channel order
- Frame extraction: decoded on the fly, or pre-extracted to disk? Which library?
- Are face crops used? Is there face detection/alignment? Which detector?
- **Are facial landmarks computed or stored anywhere?**
- Clip sampling: sequence length, stride, random vs centre, train vs eval difference
- Normalization statistics and resize/crop pipeline
- Train/val/test split logic: where implemented, is it identity-aware, how are
  identities derived from filenames?
- Class balancing: oversampling, weighted sampler, weighted loss — where applied,
  and to which splits?

### B. Models
- Every architecture class, file, and how backbones are instantiated (timm? torchvision?)
- How temporal heads attach to backbones — the exact interface/contract
- List every head variant implemented (mean/sequence, LSTM, TCN, Transformer,
  ConvLSTM, hybrid variants) and where each lives
- Are backbones frozen or fully fine-tuned?

### C. Training infrastructure
- Entry-point scripts and how they're invoked (CLI args? config files?)
- The config system: format, key names, defaults, precedence
- Training loop: optimizer, scheduler, AMP, EMA, gradient clipping, early stopping
- Checkpoint policy: what is saved, when, selection criterion
- Logging: what's written where (CSV? TensorBoard? JSON?)
- Seed handling and determinism

### D. Evaluation & metrics
- Which metrics are computed and in which file
- **For F1: which class is treated as positive?** Quote the line.
- Is evaluation frame-level, clip-level, or video-level? Where is aggregation done?
- Are per-sample predictions/probabilities exported? To what format?
- Threshold selection: fixed 0.5, or tuned? On which split?

### E. Experiment inventory
Table of every completed run discoverable from checkpoints, logs, or result files:
run ID, category, backbone, head, dataset scope, loss, and any recorded metrics.
Mark clearly whether each metric was **read from a file** or `[INFERRED]`.

### F. Documentation
For every `.md`: path, purpose, and whether its claims still match the code.
**Explicitly flag contradictions between docs and code.**

---

## PHASE 4 — REDUNDANCY & DEAD-CODE AUDIT

This section matters most. Two development cycles have overlapped in this repo.

Identify and categorize:

1. **Duplicate implementations** — the same functionality implemented in 2+ places.
   Give both paths and state which appears current (justify: imports, recency, git).
2. **Superseded directories** — anything named `*_old`, `temp/`, `backup/`, `v1/`, or
   functioning as an archive. State what supersedes it.
3. **Orphaned modules** — `.py` files imported by nothing and not an entry point.
   Verify with a repo-wide import grep before claiming this.
4. **Dead artifacts** — `__pycache__`, stale checkpoints from abandoned runs, duplicate
   figure sets, `.pyc` leftovers. Give total reclaimable size.
5. **Divergent copies** — near-identical files that have drifted. Summarize the diff.
6. **Hardcoded paths and machine-specific assumptions** — absolute paths, drive letters,
   OS-specific separators, user directories.
7. **Config sprawl** — the same parameter defined in multiple places with different values.

For each item output: `path` · `category` · `evidence` · `recommendation`
(one of KEEP / ARCHIVE / DELETE / MERGE) · `confidence` (high/medium/low).

**Recommend only. Delete nothing.**

---

## PHASE 5 — REUSABILITY ASSESSMENT

The next development cycle needs to:
- rebuild the corpus with identity-disjoint splits and asymmetric clip sampling
- pre-extract face crops **with landmark sidecars**
- support 16-frame clips at stride 4 and multi-clip inference
- add self-blended-image (SBI) augmentation as a dataset-level wrapper
- swap in a new metrics module (video-level AUC primary)
- run multi-seed experiments with aggregated statistics

For each of those six needs, state:
- which existing code can be reused as-is
- which needs modification (and roughly what)
- which must be written fresh
- the exact integration point (file, class, function signature) a new module would
  attach to

---

## OUTPUT SPEC — `Current-Status.md`

Use this exact section order:

```
1.  Executive Summary            (<= 400 words: what this repo is, what state it's in,
                                  the 5 most important findings)
2.  Repository Map               (annotated tree; one line per significant dir)
3.  Environment & Dependencies   (imports actually used; requirements files; versions)
4.  Python Module Inventory      (table: path | LOC | purpose | entry point? | imported by)
5.  Data Pipeline                (per Phase 3A, with signatures quoted verbatim)
6.  Model Zoo                    (per Phase 3B)
7.  Training Infrastructure      (per Phase 3C)
8.  Evaluation & Metrics         (per Phase 3D)
9.  Experiment Inventory         (per Phase 3E)
10. Documentation Inventory      (per Phase 3F)
11. Redundancy & Dead Code       (per Phase 4, sorted by reclaimable size desc)
12. Technical Debt & Risks       (ranked; what would break a rewrite)
13. Reusability Assessment       (per Phase 5)
14. Open Questions               (every [UNKNOWN], with how to resolve it)
```

Formatting rules:
- Tables wherever the content is tabular.
- Fenced code blocks for signatures and config excerpts, quoted verbatim.
- `path:line` references for every non-obvious claim.
- Tag inferences `[INFERRED]` and gaps `[UNKNOWN]` inline.
- No praise, no filler, no speculation about intent. Describe what exists.
- If the repo contradicts itself, say so plainly and show both sides.

Begin with Phase 1 and work through in order. Write `Current-Status.md` only after
all phases are complete.
