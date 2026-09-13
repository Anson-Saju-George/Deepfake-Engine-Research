# Reading List — real-vs-fake video detection (whole-frame, cross-fake-type)

Companion to `REFERENCES.bib`. Tiers reflect how much of each paper to actually read,
not how important the citation is.

- **Tier 1** — read properly, cover to cover. These shape the framing.
- **Tier 2** — read method + results, skim the rest. You need their numbers and protocols.
- **Tier 3** — abstract + cite.

---

> **REVISED Sept 2026 after a second literature check.** Six papers were added and the
> tiers reordered. The novelty claim narrowed: UNITE (CVPR 2025) already demonstrates
> unified whole-frame detection, and Omni-Fake (CVPR 2026) already formalises the
> real / partially-manipulated / fully-synthetic taxonomy. What remains open is the
> **controlled bidirectional cross-regime transfer experiment under audited conditions.**
> Read the four ⚡ papers below FIRST — they define what is already taken.

## Tier 1 — read properly (12)

### ⚡ Read these four first — they define the remaining novelty

| # | Title | Authors, venue | Why |
|---|---|---|---|
| 0a | **Towards a Universal Synthetic Video Detector (UNITE)** | Kundu et al., CVPR 2025, pp. 28050–28060 | **The closest prior work.** Whole-frame, no face crop, spans face-manip + T2V/I2V, beats face-crop SOTA (FF++ 99.96, CelebDF 95.11). Validates your pipeline choice. But it trains FF++ vs FF++ + GTA-V *game footage*, not AI-generated video, and was **not** trained on the GenVideo split — so the bidirectional decomposition is still open. Its FF++→DeMamba 57.38 vs FF++GTA-V→DeMamba 93.75 is the single most relevant number in the literature to your RQ1. |
| 0b | **Auditing Generalization in AI-Generated Video Detection (VidAudit)** | Cakiroglu, Lu, Dalkilic & Kurban, arXiv 2606.31004, 2026 | **Adopt this protocol; do not claim its controls as your own.** A three-feature *clip-length* classifier hits LOGO AUC 0.998 unaudited → 0.529 audited. A 20-paper survey found none applying all six controls. A CLIP baseline was caught carrying dataset identity. Also changes your metrics: at FPR 0.1% high-AUC methods collapse to single-digit recall, so they recommend an audited **tuple** (AUC, above-floor margin, operating-point recall, calibration). |
| 0c | **DeMamba / GenVideo** | Chen et al., arXiv 2405.19707 (also Sci. China Inf. Sci.) | **Your synthetic-regime training corpus.** 1,223,511 real + 1,078,838 fake; OOD test 10,000 real + 8,588 fake from different generators *and* different real sources. UNITE benchmarks against it, so using it makes you comparable. Caveat: train reals Kinetics-400 + Youku-mPLUG, test reals MSR-VTT. |
| 0d | **Omni-Fake** | CVPR 2026 *(verify authors)* | Formalises real / partially-manipulated / fully-synthetic as a ternary problem (~100k real, 100k synthetic, 10k tampered video). **This removes "treat them as distinct categories" from your novelty.** You must position against it. |

### The original eight



| # | Title | Authors, venue | Why |
|---|---|---|---|
| 1 | **Shortcut Learning in Deep Neural Networks** | Geirhos et al., Nature MI 2020 | The single biggest methodological risk. Whole-frame + heterogeneous sources means the model can learn dataset signature instead of fakery. Gives you the vocabulary and diagnostics to address it before a reviewer does. **Read this first — it reframes everything after it.** |
| 2 | **Detecting Deepfakes with Self-Blended Images** | Shiohara & Yamasaki, CVPR 2022 | Must justify NOT using it. Strongest cross-dataset result in the field (92.87 CDF AUC) and it's face-based, so "we went whole-frame, therefore SBI is unavailable" has to be an argued position, not an omission. |
| 3 | **Human Action CLIPs: Detecting AI-generated Human Motion** | Bohacek & Farid, arXiv 2412.00526 | Your synthetic test set (DeepAction), and the closest existing work to the cross-fake-type question. Note *how* they evaluate — CLIP embeddings, not pixel CNNs. That contrast matters. |
| 4 | **DeepSpeak Dataset v1.0** | Barrington, Bohacek & Farid, arXiv 2408.05366 | Table 1 is your dataset-landscape justification. Also states explicitly that SOTA detectors fail to generalize to modern generators. |
| 5 | **FaceForensics++: Learning to Detect Manipulated Facial Images** | Rössler et al., ICCV 2019 | You train on it. Know the splits, compression levels, and the baseline numbers you'll be compared against. |
| 6 | **Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics** | Li et al., CVPR 2020 | Your canonical cross-dataset test. Know the official test list and why it exists. |
| 7 | **Community Forensics: Using Thousands of Generators to Train Fake Image Detectors** | Park & Owens, CVPR 2025 | Closest thing to your thesis — training across many generators for generalization. Directly informs whether pooling fake types helps or hurts. |
| 8 | **Unbiased Look at Dataset Bias** | Torralba & Efros, CVPR 2011 | Fourteen years old, still the sharpest statement of your problem. The "name that dataset" experiment is exactly the sanity probe to run. |

**Papers 1, 3, 7, 8 let you interpret a *failure* to transfer as a finding rather than a
bug. Papers 0a–0d define what is already claimed, and therefore what is left.**

---

## Tier 2 — method + results, skim the rest (7)

| # | Title | Authors, venue |
|---|---|---|
| 9 | **Face X-ray for More General Face Forgery Detection** | Li et al., CVPR 2020 |
| 10 | **Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection** | Haliassos et al., CVPR 2021 |
| 11 | **End-to-End Reconstruction-Classification Learning for Face Forgery Detection** (RECCE) | Cao et al., CVPR 2022 |
| 12 | **Exploring Temporal Coherence for More General Video Face Forgery Detection** (FTCN) | Zheng et al., ICCV 2021 |
| 13 | **DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection** | Jiang et al., CVPR 2020 |
| 14 | **ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis** | He et al., CVPR 2021 (Oral) |
| 15 | **DF40: Toward Next-Generation Deepfake Detection** | Yan et al., NeurIPS 2024 |

---

## Tier 3 — abstract + cite (13)

### Backbone architectures
| # | Title | Authors, venue |
|---|---|---|
| 16 | **A ConvNet for the 2020s** (ConvNeXt) | Liu et al., CVPR 2022 |
| 17 | **An Image is Worth 16x16 Words** (ViT) | Dosovitskiy et al., ICLR 2021 |
| 18 | **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** | Liu et al., ICCV 2021 |
| 19 | **Xception: Deep Learning with Depthwise Separable Convolutions** | Chollet, CVPR 2017 |
| 20 | **EfficientNet: Rethinking Model Scaling for CNNs** | Tan & Le, ICML 2019 |

### Video architectures
| # | Title | Authors, venue |
|---|---|---|
| 21 | **X3D: Expanding Architectures for Efficient Video Recognition** | Feichtenhofer, CVPR 2020 |
| 22 | **SlowFast Networks for Video Recognition** | Feichtenhofer et al., ICCV 2019 |
| 23 | **TSM: Temporal Shift Module for Efficient Video Understanding** | Lin et al., ICCV 2019 |

### Frequency-domain methods
| # | Title | Authors, venue |
|---|---|---|
| 24 | **Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues** (F3-Net) | Qian et al., ECCV 2020 |
| 25 | **Spatial-Phase Shallow Learning** (SPSL) | Liu et al., CVPR 2021 |
| 26 | **Multi-Attentional Deepfake Detection** | Zhao et al., CVPR 2021 |

### Datasets / surveys
| # | Title | Authors, venue |
|---|---|---|
| 27 | **The DeepFake Detection Challenge (DFDC) Dataset** | Dolhansky et al., arXiv 2020 |
| 28 | **MVFNet: Multipurpose Video Forensics Network** | Nguyen & Stamm, WACV 2025 |

**Lowest priority (in the bib, cite only if needed):** DFDC-Preview, DeepFake-TIMIT,
Recognition in Terra Incognita, Corvi et al. (synthetic video), and the two surveys.

---

## Before the new chat — three actions

1. **Verify the `[VERIFY pages]` entries in REFERENCES.bib.** DBLP (`dblp.org`) has clean
   BibTeX for every CVPR/ICCV/ECCV/NeurIPS paper listed. ~20 minutes, and it removes the
   risk of submitting a fabricated page range.
2. **Read Tier 1 in the order given.** Geirhos first is deliberate.
3. **Search fresh for 2025–26 synthetic-video detection work.** That literature is moving
   fast and is the one genuinely thin section of the bib.

---

## Cut from the old manuscript — do not reinstate

Roughly **12 of the original 39 references were irrelevant**. Reasons recorded so the
decision is not re-litigated:

| Cut | Reason |
|---|---|
| 6 medical-imaging refs (medical segmentation, hyperspectral reconstruction, medical image computing) | Zero connection to deepfake detection. Pure padding. |
| 4 generic vision-architecture refs (MambaVision, LAVIN-DiT, SpectFormer, HTR-ViT) | Not deepfake work. |
| 2 unrelated-forensics refs (cloud authentication via facial forensics, copy-move forgery) | Different problem entirely. |
| Assorted transformer surveys | One survey is enough, not four. |

A padded bibliography signals a weak literature review — and this manuscript already has
one rejection.
