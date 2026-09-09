![DeepFake Detection Research Pipeline Banner](images/Banner.png)

# DeepFake Detection Research Pipeline

> Research pipeline for **video authenticity detection across two forensic regimes** —
> face manipulation and fully synthetic video — with protocol-aware data loading,
> dataset-audit tooling, whole-frame methodology, and reproducible experiment tracking.

## What this project is now

The scientific goal is **cross-regime generalization**, not "another universal detector."
A video is either authentic or produced/altered by a generative model, and that spans two
technically distinct phenomena:

| Regime | What "fake" means | Where the evidence lives | Train corpus |
|---|---|---|---|
| **Face manipulation** | A real recording with the face swapped/reenacted | A local blending seam / warp | FF++ (DF + DFD subset) |
| **Fully synthetic** | The entire video generated from a prompt | Whole-frame: physics, texture, temporal incoherence | GenVideo |

The contribution is a progression of three questions — **cross-regime transfer (and its
asymmetry) → genuine unification vs. two decision regions → survival under shortcut audits.**

### Authoritative docs (read these first)

| Doc | Role |
|---|---|
| **`PROJECT_CONTEXT.md`** (root) | whole-project scope, layout, caveats — start here |
| **`papers/writing/PROJECT_VISION.md`** | why the problem matters (the vision) |
| **`papers/writing/RESEARCH_QUESTIONS.md`** | the three RQs (the contribution) |
| **`papers/writing/EXPERIMENT_DESIGN.md`** | the frozen experimental design (the *how*) — **authoritative** |
| **`papers/writing/READING_LIST.md`** + `REFERENCES.bib` | tiered literature |
| **`perf/PERFORMANCE_LOG.md`** (+ `PERFORMANCE_CONTEXT.md`) | all measured performance numbers |

> **Note on framing.** The image-vs-video modality framing that earlier versions of this repo
> were built around is **superseded** by `EXPERIMENT_DESIGN.md`. The 22 completed image/video
> runs are **archived baseline evidence**, not the new paper's contribution (see below).

## Design decisions (new direction)

- **Whole-frame detection** — no face detection, no face cropping (the old face-crop pipeline is legacy).
- **Two backbones**, not a zoo: **ConvNeXt-B** + a foundation encoder (**SigLIP-So400M** / DINOv2).
- **Regime as the independent variable**, not architecture — the old temporal-head sweep is cut.
- **Audited metrics**: AUROC + above-floor margin + recall@0.1%FPR + calibration, reported **per-dataset, never pooled**, with multi-seed + significance testing.
- **Shortcut audits**: VidAudit-style controls + a 32×32 thumbnail diagnostic (RQ3).

## Datasets

On disk (loose image jpgs read directly by the loader; videos as `.mp4`):

- **Images:** `cifake`, `ai-generated-images-vs-real-images`
- **Videos:** `celeb-df-v2`, `faceforensics++` (DF + DFD subset — see the honesty note below), `real-ai-videos`, `deepaction-v1`

**Canonical registry** — sizes, real/fake counts, access status, planned pulls, and rejected
candidates — is **`proc/dataset_downloader/DATASETS.md`** (with the action plan in `TODO.md`).

> **⚠ Naming honesty:** the on-disk `faceforensics++/` is **`Deepfakes` + Google/Jigsaw
> `DeepFakeDetection` (DFD)** + reals — **not** the full 4-manipulation FF++. Cite both sources;
> don't write "trained on FaceForensics++" implying DF/F2F/FS/NT. Full note in `DATASETS.md`.

## Dataloader truth (`data/dataloader.py`)

The active, protocol-aware loader is still the data backbone. Label convention is
**`real = 1`, `fake = 0`** across every dataset.

Protocols and sampling:

- `image_only` — image-domain spatial learning (source train/test boundaries preserved; val carved from train).
- `video_only` — raw-video learning; identity-aware grouped splitting.
- `frame_only` / `combined_aux` — derived-frame and auxiliary mixed-media studies.
- `mode="single"` — one sampled frame; `mode="sequence"` — contiguous clip (`seq_len`, `stride`). Train = random offset, eval = centered.

Split/leakage/identity rules live in `data/splits.py` + `data/identity.py`; low-level
label/split conventions were consolidated into `PROJECT_CONTEXT.md` and `DATASETS.md`.

## Active code layout

| Path | What |
|---|---|
| `data/` | protocol-aware DataLoader, splits, identity, cache tools |
| `train/image/` | active image trainer, registry (`image_models.py`), eval (`test_image.py`) |
| `train/` | `eval_predictions_common.py`, `evaluate_all.py` (video path optional — video trainer archived) |
| `proc/dataset_downloader/` | dataset registry + download tooling |
| `proc/code_snippets/` | reference implementations (metrics, splits, SBI, perturbations, complexity profiler) |
| `graphs/` | figure generator + rendered PNGs |
| `books/` | lifecycle chapters (01–12) — paper-writing context (see its direction banner) |
| `perf/` · `papers/` | performance record · the paper (writing/ + literature/) |
| `temp/` | archived legacy (superseded trainers, old docs, old checkpoints) — not active |

## Archived baseline (old framing)

The completed **22 runs** (5 image, 17 video) live as archived evidence. Best image:
`IMG-EXP-04` ConvNeXt-Base (F1 `0.9863`); best video: `VID-TMP-02` ConvNeXt-Large sequence
(F1 `0.7841`, acc `0.9089`). Full tables + the archived experiment surface are in
`books/08_model_evaluation.md`; the video trainer tree is archived under
`temp/legacy_models/train_video/`. These are a face-centric baseline, **not** the contribution.

## First commands

```bash
# Audit dataset truth
python -m data.dataset_analyzer

# Validate image / raw-video readability
python -m data.dataset_run --dtype image --num-workers 16
python -m data.dataset_run --dtype video --num-workers 8

# Corpus + split distribution the pipeline would use
python train_data_pipeline_pull.py --mode video --datasets celeb-df-v2 faceforensics++

# Image baseline (archived-framing example; still runnable)
python -m train.image.run_image --exp IMG-EXP-01 --dataset-scope image_combined
python -m train.image.test_image --workers 8 --prefetch-factor 4 --batch-size 128
```

> The new whole-frame extractor and cross-regime training entrypoints described in
> `EXPERIMENT_DESIGN.md` are the next code to be written; the commands above exercise the
> existing (image-side) pipeline and the dataset-audit tooling.
