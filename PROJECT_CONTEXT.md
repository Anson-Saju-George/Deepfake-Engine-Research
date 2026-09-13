# Project Context & Scope — read this first

Orientation for anyone (human or agent) opening this repo cold. It explains **what the
project is**, **the current research direction**, **the datasets**, and **where everything
lives**. For performance/benchmark context specifically, see `perf/PERFORMANCE_CONTEXT.md`
(and `perf/PERFORMANCE_LOG.md` for the numbers).

---

## 1. What this project is

A **deepfake-detection research pipeline** (image + video) built on `timm` backbones with a
custom protocol-aware DataLoader (`data/dataloader.py`). It has already produced **22
completed training runs** (5 image, 17 video) under the *old* research framing, and is now
being **re-oriented** toward a new thesis (§2). The repo doubles as the writing base for a
research paper — the `books/` directory is the structured context a paper-writing agent reads.

---

## 2. Current research direction — AUTHORITATIVE

**`papers/writing/01_START_HERE/EXPERIMENT_DESIGN.md` is the single authoritative design.** It supersedes the
older image-vs-video modality framing.

> **New thesis:** Face manipulation and fully synthetic video are related but non-identical
> forensic regimes. Quantify **bidirectional** cross-regime transfer (train-on-one /
> test-on-the-other), test whether balanced joint training yields regime-invariant
> evidence, and audit against dataset-specific shortcuts.

Consequences that override older docs:
- **Whole-frame detection, no face detection / no face cropping** (older face-crop caching is legacy).
- Two backbones only: **ConvNeXt-B** + a foundation encoder (**SigLIP-So400M** / DINOv2). The temporal-head sweep (TCN/LSTM/Transformer/ConvLSTM) is **CUT** — the independent variable is training *regime*, not architecture.
- **GenVideo** is the synthetic training corpus; **DeepAction** returns to test-only; **ForgeryNet dropped** from core.
- Audited metric tuple (AUROC + above-floor margin + recall@0.1%FPR + calibration), per-dataset never pooled, multi-seed + Wilcoxon.
- The **old 22 runs are archived, not designed around** (at most 1–2 reruns as a face-centric baseline).

Companion docs: `papers/writing/02_references/READING_LIST.md` (tiered literature), `papers/writing/02_references/REFERENCES.bib`,
`papers/writing/03_dataset_intel/DATASETS_FINAL.md`, `papers/writing/06_superseded/REVISION_AUDIT.md` (prior reviewer response).

---

## 3. Datasets

On disk now (post-cleanup): `datasets/images/{cifake, ai-generated-images-vs-real-images}`
(loose jpgs, read directly by the loader — do **not** archive them), `datasets/videos/
{celeb-df-v2, faceforensics++, real-ai-videos, deepaction-v1}`. Full registry with sizes,
fake/real counts, access status, and rejected candidates: **`proc/dataset_downloader/DATASETS.md`**.

---

## 4. Where things live (post-cleanup layout)

| Path | What |
|---|---|
| `data/` | active DataLoader, splits, cache, dataset tools |
| `train/image/` | active image trainer + eval |
| `train/` (common, evaluate_all) | shared eval; video path optional (video trainer archived) |
| `proc/dataset_downloader/` | dataset registry (`DATASETS.md`) + download tooling |
| `proc/code_snippets/` | reference implementations (metrics, splits, SBI, perturbations, complexity profiler) — relocated from the old `docs/code/` |
| `perf/` | `PERFORMANCE_LOG.md` (numbers), `PERFORMANCE_CONTEXT.md` (how to read them), `CACHE_PERFORMANCE.md` |
| `papers/writing/` | consolidated handoff bundle (read top-to-bottom): `README.md` index → `01_START_HERE/` (EXPERIMENT_DESIGN [authoritative], PROJECT_VISION, HANDOFF_PROMPT, RESEARCH_QUESTIONS) → `02_references/` (READING_LIST, REFERENCES.bib/.md) → `03_dataset_intel/` (DATASET_DECISIONS, FFPP_*, FORGERYNET_BRIEF, DATASETS_FINAL) → `05_provenance_prompts/` → `06_superseded/` (RESEARCH_PLAN, REVISION_AUDIT, revised docx). Plus `Deepfake-AI-Conversation.pdf` (Claude+ChatGPT session export). |
| `papers/literature/` | 42 reference PDFs, tiered: `tier1_core/`, `tier2_methods/`, `tier3_reference/` |
| `books/` | structured lifecycle chapters (01–12) — the paper-writing context |
| `graphs/` | figure generator + rendered PNGs from the completed runs |
| `temp/` | legacy checkpoints, superseded trainers, archived docs (`temp/legacy_docs/`, `temp/legacy_models/`) — archive, not active |

> **Note:** the old `test/` (A–E benchmark harness) and `docs/` directories were removed
> after their content was consolidated into `perf/` (performance) and `books/` + `papers/`
> (research). Superseded-but-detailed docs were archived to `temp/legacy_docs/`.

---

## 5. History / caveats worth knowing

- Old **temporal vs spatiotemporal** runs share one trainer class — `VID-TMP-02` and `VID-ST-03` are numerically identical, **not** two distinct paradigms. The new design removes this ambiguity.
- **Label convention (code-truth, preserved from the retired CONVENTIONS.md):** raw label values are **`real = 1`, `fake = 0`** for every dataset (set in `data/dataloader.py:DATASET_CONFIG`). This does not change.
- Old code had an **F1 positive-class inconsistency** (active eval `train/eval_predictions_common.py`: label 1 = real is positive; reference `proc/code_snippets/metrics.py`: label 1 = fake). Accuracy/AUC are symmetric; **F1/precision/recall are not — always state which class is positive** when quoting old F1s. (Split/leakage rules and identity handling live in `data/splits.py` + `data/identity.py`.)
- All 22 runs used **seed=42** — no multi-seed/significance is derivable from existing artifacts (a reviewer objection the new design fixes).
