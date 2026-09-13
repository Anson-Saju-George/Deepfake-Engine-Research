# DATASETS_FINAL

## Deepfake Detection Dataset Landscape and Final Selection

This document records the complete dataset landscape considered for the project, the datasets selected for the current paper, their intended roles, split policies, and the provenance caveats that must be resolved before final reporting.

---

# Table 1 — All datasets considered

| Dataset | Real | Fake | Total | % Real | Size | Status |
|---|---:|---:|---:|---:|---:|---|
| **FF-derived (DF+DFD)** ✅ on disk | 1,363 | 4,066 | 5,429 | 25% | 26.1 GB | **SELECTED — train** |
| **Celeb-DF v2** ✅ on disk | 590 | 5,639 | 6,229* | 9.5% | 9.5 GB | **SELECTED — test** |
| **Real-AI videos** ✅ on disk | 33 | 33 | 66 | 50% | 2.65 GB | **SELECTED — test (marginal)** |
| **DFDC-Preview** | 1,131 | 4,119 | 5,250 | 22% | ~20 GB | **SELECTED — test (to download)** |
| DeeperForensics-1.0 | 48,475 | 11,000 | 59,475 | 82% | ~100+ GB | Phase 6 — test |
| KoDF | 62,166 | 175,776 | 237,942 | 26% | ~1 TB | Phase 6 — test |
| DeepSpeak v1/v2 | ~8k clips | ~6k clips | ~14k+ | ~50% | ~40+ GB | Phase 6 — test |
| FF++ full (DF/F2F/FS/NT) | 1,000 | 4,000 | 5,000 | 20% | ~40 GB c23 | ⚠️ only partial on disk |
| DFDC (full) | 23,654 | 104,500 | 128,154 | 18% | ~470 GB | ❌ rejected — size |
| ForgeryNet (video) | ~99,630 | 121,617 | 221,247 | 45% | ~500 GB | ❌ rejected — comparability |
| UADFV | 49 | 49 | 98 | 50% | tiny | ❌ rejected — too small / old |
| DFD (Google, standalone) | 363 | 3,068 | 3,431 | 11% | — | Already represented in current DFD subset |
| CIFAKE (image) | 60,000 | 60,000 | 120,000 | 50% | — | Drop / de-emphasize |
| AI-vs-Real (image) | 29,990 | 29,998 | 59,988 | 50% | — | Keep as image reference |

\* **Celeb-DF discrepancy:** the published real/fake totals sum to 6,229, while the current frame-profile count reports 6,533 videos. The ~300-video difference is likely related to the YouTube-real subset, but this must be verified before the number appears in the manuscript.

### Provenance

- ✅ **On-disk rows:**
  - FF-derived counts are measured from the current Phase 0 corpus.
  - Celeb-DF and Real-AI counts currently come from documentation and still require disk verification.
- **Non-acquired datasets:** counts are taken from published papers or official documentation, not from local measurement.
- Exact counts must be re-established locally if any of these datasets are later acquired.

---

# Table 2 — Selected datasets and split policy

| Dataset | Role | Real | Fake | % Real | Split | Frames | Cache @ q90 |
|---|---|---:|---:|---:|---|---:|---:|
| **FF-derived (DF+DFD)** | **TRAIN** | 1,363 | 4,066 | 25% | 720 / 140 / 140 identity-disjoint → clip-balanced ~50% for train | 3,573,959 | ~40 GB |
| **Celeb-DF v2** | **TEST** | 590 | 5,639 | 9.5% | Official 517-video test list (178 real / 340 fake) | 2,476,312 | ~24 GB |
| **DFDC-Preview** | **TEST** | 1,131 | 4,119 | 22% | Own test partition | — not downloaded | — |
| **Real-AI** | **TEST (marginal)** | 33 | 33 | 50% | Per-file, entire set used for evaluation | 28,831 | ~0.4 GB |

---

# Split mechanics

## FF-derived training corpus

The current FF-derived training corpus uses the official FF++ split structure:

- **720 identities / source groups for training**
- **140 for validation**
- **140 for test**

Identity and source grouping are enforced through filename-derived grouping and union-find based disjoint assignment.

### Class balance

At video level:

- Real: ~25%
- Fake: ~75%

Training clips are balanced through **asymmetric clip sampling**:

- approximately **3 clips per real video**
- approximately **1 clip per fake video**

This moves the training distribution toward approximately **50 / 50 at clip level** without discarding any videos.

Validation and test sets retain their natural class proportions.

---

## Celeb-DF v2

Celeb-DF is treated as **evaluation-only**.

Use the official:

```text
List_of_testing_videos.txt
```

The documented official test list contains approximately:

- **178 real**
- **340 fake**

No Celeb-DF sample may be used for:

- model training;
- threshold tuning;
- augmentation selection;
- checkpoint selection;
- hyperparameter search.

---

## DFDC-Preview

DFDC-Preview is designated as **test-only**.

It should be evaluated using its native partitioning and must remain completely isolated from training and model selection.

---

## Real-AI videos

The Real-AI collection is very small:

- 33 real
- 33 fake

Because of its size, the entire collection is used as a marginal external evaluation set rather than being split again.

Results from this dataset should be interpreted cautiously and should not carry a major manuscript claim on their own.

---

# LOMO — Cross-manipulation protocol

The compressed journal-paper plan uses two leave-one-manipulation-out style comparisons:

1. **Train on Deepfakes → test on DeepFakeDetection**
2. **Train on DeepFakeDetection → test on Deepfakes**

These experiments test whether the learned detector transfers between two distinct manipulation pipelines.

For the full thesis, this should later expand to all available FF++ manipulation families:

- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures

---

# Dataset role policy

Before any experiment begins, every dataset must have a fixed role.

Recommended labels:

```text
TRAIN
VALIDATION
IN-DOMAIN TEST
CROSS-MANIPULATION TEST
CROSS-DATASET TEST
ROBUSTNESS TEST
TEMPORAL-DECAY TEST
MULTIMODAL TEST
```

Once a dataset is marked **test-only**, it must not influence:

- model selection;
- augmentation design;
- checkpoint selection;
- threshold tuning;
- early stopping;
- architecture choice based on its performance.

---

# Provenance caveats to resolve before manuscript freeze

## 1. Celeb-DF and Real-AI local counts

The current FF-derived counts:

```text
1,363 real
4,066 fake
```

are locally measured.

The current Celeb-DF and Real-AI real/fake totals are still documentation-derived and must be verified against disk before final tables are generated.

---

## 2. Published numbers for non-acquired datasets

The following datasets have not yet been locally acquired:

- DeeperForensics-1.0
- KoDF
- DeepSpeak
- full DFDC
- ForgeryNet

Their current counts in this document are **literature-level landscape figures**, not local measurements.

If any of these datasets enter the actual experimental corpus, replace the literature counts with:

- locally verified file counts;
- decode-valid video counts;
- usable face-track counts;
- final manifest counts.

---

# Current dataset decision

## Selected for the journal paper

### Training
- **FF-derived (DF + DFD)**

### Primary external testing
- **Celeb-DF v2**
- **DFDC-Preview**

### Secondary / marginal external testing
- **Real-AI videos**

---

## Deferred to thesis phases

- DeeperForensics-1.0
- KoDF
- DeepSpeak
- full FF++
- larger modern datasets
- audio-visual datasets

---

## Rejected for the current paper

### DFDC full
Reason: storage and compute cost are disproportionate to the immediate paper objective.

### ForgeryNet
Reason: excellent scale and diversity, but lower priority for the current comparability-focused journal paper.

### UADFV
Reason: too small and historically dated to add meaningful evidence.

### CIFAKE
Reason: image-generation task differs from the core face-video protocol and would expand the manuscript scope unnecessarily.

---

# Final working principle

> **The paper should train on a controlled, reproducible source corpus and test generalization on datasets that remain strictly unseen during training.**

The goal is not to maximize the number of datasets in one table. The goal is to make each dataset serve a clearly defined scientific role.
