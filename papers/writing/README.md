# Handoff Bundle — Cross-Regime Video Authenticity Detection

Everything carried forward from the design conversation (Jun–Sep 2026). **Read this file
first** — a fair amount of what's here is superseded, and knowing which is which matters
more than reading all of it.

---

## If you read only three files

| Order | File | Why |
|---|---|---|
| 1 | `01_START_HERE/HANDOFF_PROMPT.md` | **Paste this to start the new chat.** Role, decided-vs-open, guardrails, first task |
| 2 | `01_START_HERE/PROJECT_VISION.md` | The problem, the research question, the full technical approach |
| 3 | `01_START_HERE/EXPERIMENT_DESIGN.md` | Frozen plan: dataset roles, three training conditions, metrics, audits |

Everything else is reference or provenance.

---

## The one-paragraph version

A paper on face-manipulation deepfake detection was **rejected by TCSST in July 2026** for
redundancy, no generalization evaluation, and no statistical rigour. The root cause of the
second was structural: it trained on Celeb-DF and FF++ *combined*, so no genuine
cross-dataset test existed. The project was then rebuilt around a different question —
**does forensic evidence transfer between face manipulation and fully synthetic video, in
either direction?** — with a whole-frame pipeline (no face cropping), one training source
per regime, and an audited evaluation protocol. Infrastructure is built and benchmarked.
The experiments have not started. The critical path is `extract_frames.py`, which does not
yet exist.

---

## Folder guide

### `01_START_HERE/` — ✅ current
The three documents above. Authoritative.

### `02_references/` — ✅ current
- `REFERENCES.bib` — 44 entries, sectioned, arXiv entries in proper `eprint` form.
  `[VERIFY]` markers flag citation details needing a DBLP check before submission.
- `REFERENCES.md` — annotated version with the *numbers* that matter (cross-dataset AUC
  ranges, UNITE's two decisive rows, SBI's 92.87 on Celeb-DF)
- `READING_LIST.md` — 28 papers in three tiers. The four ⚡ papers define what's already
  claimed and therefore what's left.

**Replace any existing `docs/REFERENCES.bib` with this one** — do not merge, there are key
collisions with an earlier 4-entry version.

### `03_dataset_intel/` — 🟡 mostly current, one caveat
- `FFPP_DOWNLOAD.md`, `FFPP_SCRIPT_CHECK.md`, `FFPP_VARIANTS_CHECK.md` — ✅ **still
  actionable.** FF++ F2F/FS/NT (4.63 GB, c23) is the one download that unblocks training.
  Local download script verified working.
- `FORGERYNET_BRIEF.md` — full intel (496.23 GB, dual-MD5 problem, OpenDataLab route).
  ForgeryNet is **dropped from the core experiments** — keep this only if it's revisited.
- `DATASET_DECISIONS.md` — ⚠️ **partially superseded.** Written before the pivot. It still
  reflects face-crop-era roles and predates GenVideo. `EXPERIMENT_DESIGN.md` §1 overrides it.

### `04_code/` — 🟡 three of four modules survive the pivot
| Module | Status |
|---|---|
| `metrics.py` | ✅ **keep** — label-convention boundary (`repo_labels=True`), AUC-primary, both F1 conventions, bootstrap CI, seed summary, Wilcoxon |
| `splits.py` | ✅ **keep** — identity-disjoint splits, leakage checker, asymmetric clip sampling, LOMO configs |
| `perturbations.py` | ✅ **keep** — 7 distortion types × 5 levels; robustness is still in the plan |
| `sbi.py` | ❌ **superseded** — self-blending needs face landmarks; the pipeline is now whole-frame. Kept for the record only |

### `05_provenance_prompts/` — 📁 archive
The prompts that produced the engineering findings. **The findings themselves are already
baked into `EXPERIMENT_DESIGN.md`** — these exist so a result can be traced to how it was
measured. Don't re-run them.

### `06_superseded/` — ⚠️ historical, do not build on
- `Deepfake_Research_Revised.docx` — **the rejected manuscript.** Its protocol
  (face crops, combined-corpus training, minority-class F1) is superseded. Read for what
  *not* to repeat.
- `RESEARCH_PLAN.md` — the earlier Tier A/B/C phased plan, built around SBI and face crops
- `sbi_demo.png` — SBI blend/mask visualisation

---

## Engineering findings already settled (don't re-derive)

| Finding | Value |
|---|---|
| Environment | `/opt/ml` venv · torch 2.11.0+cu129 · cuDNN 92500 · RTX 5080 16.3 GB · verified clean |
| Decode | **pyav** 7.5× faster than cv2, pixel-identical |
| Dataloader | `num_workers=12` (plateaus there) |
| Precision | **AMP bf16** +56% throughput, ~42% less VRAM. fp16 within 3.5% but needs loss scaling |
| Cache location | **F:** (drvfs) — dataloader wait **0.25 ms**, GPU-bound at 76–86% util |
| Throughput | seq16/batch4 ≈ **13.56 clips/sec** |
| ⚠️ VRAM | **batch ≥12 silently spills to system RAM** — hard cap at 4 |
| Rejected | NVDEC decode (slower + correctness bugs) · Intel iGPU decode (no `/dev/dri`) · `torch.compile` (11× slower) · `channels_last` alone without AMP (regression) |
| Resolution | **384×384** default (matches UNITE); 256/384/512 pilot to confirm |

---

## Corpus status

| Dataset | Role | State |
|---|---|---|
| FF++ (DF + reals + DFD) | **TRAIN** (face) | on disk, c23 |
| FF++ F2F/FS/NT | **TRAIN** (face) | 🟢 **4.63 GB — pull this first** |
| GenVideo | **TRAIN** (synthetic) | ❌ not acquired · ~2.26M videos, subsample to ~10–20k |
| Celeb-DF v2 | TEST | ✅ extracted + cached to F: (30.36 GB) |
| DeeperForensics-1.0 | TEST (unseen manip) | ✅ access approved · 127.5 GB manipulated-only · ⚠️ **ships no reals** |
| DeepAction v1 | TEST (cross-fake-type) | ❌ not acquired · 6.06 GB · ungated · ⚠️ all 512×512 |
| DeepSpeak v1.1/v2 | TEST (Phase 6) | 🟡 HF gated, ~1 month queue |
| DFDC | — | 🔴 blocked (AWS-only gate) |
| ForgeryNet · DF40 · KoDF · TIMIT · UADFV · Real-AI | — | ❌ rejected, reasons in `REFERENCES.md` |

---

## Guardrails from real failures in this project

1. **Plan before code.** Propose, wait, execute one task at a time.
2. **Verify before deleting.** A silent skip once dropped 16 of 22 checkpoints from an upload.
3. **Fail loudly.** A dataloader returned zero tensors with real labels on decode failure —
   29,241 logged events before it was caught.
4. **Never derive labels or identities by substring.** "fake" matched the repo folder name
   `DEEPFAKE_MODEL-DS-APP` and false-positived every record.
5. **Label convention: `real=1, fake=0`** in the data layer, converted at the metrics
   boundary. Mixing swaps F1/precision/recall while leaving AUC unchanged — invisible.
6. **DeeperForensics ships no reals** — its real class is the 1,000 FF++ YouTube targets.
   **Split by FF++ target ID first**, or the unseen-manipulation test leaks.
7. **Duration + codec normalisation before training, not as an audit.** A three-feature
   clip-length classifier reaches 0.998 AUC on a standard benchmark measuring nothing.
8. **Correctness gates speed.** A faster decoder returning different pixels is a bug.
9. **Don't churn `/opt/ml`** — verified clean, and package changes mid-experiment make runs
   non-comparable.

---

## Open decisions

1. **GenVideo extraction budget** — ~10–20k videos proposed for the controlled comparison
2. **Second backbone** — SigLIP-So400M vs DINOv2
3. **Verify Omni-Fake (CVPR 2026)** — unconfirmed, load-bearing for the novelty argument
4. **Unified-arm source control** — with each regime internally source-matched, source maps
   onto regime in the unified condition and could be learned instead of forgery

---

## Immediate next action

**Write `extract_frames.py`** — dataset-agnostic, resolution-agnostic, whole-frame, native
fps, fixed-duration clips, canonical re-encode, writing to the existing WebDataset shard
cache, with the full metadata schema in `EXPERIMENT_DESIGN.md` §7.

Without `dataset`, `regime` and `source_real_dataset` in that metadata, the shortcut audits
cannot be run at all.
