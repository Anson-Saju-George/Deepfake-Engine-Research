# References — annotated

Human-readable companion to `REFERENCES.bib`. Same entries, with the facts and numbers
that actually matter for this project. Read alongside `READING_LIST.md` (what to read) and
`EXPERIMENT_DESIGN.md` (what to run).

**Status legend:** ✅ verified by me · 🟡 from collaborator research, unconfirmed ·
⚠️ citation details need checking before submission

---

## 1. The four that reshaped the design

Read these first. They define what is already claimed, and therefore what is left.

### UNITE — Towards a Universal Synthetic Video Detector ✅
Kundu, Xiong, Mohanty, Balachandran & Roy-Chowdhury · **CVPR 2025, pp. 28050–28060** ·
arXiv:2412.12278 · UCR + Google

The closest prior work. Whole-frame, **no face cropping**, spans face manipulation *and*
fully AI-generated T2V/I2V. SigLIP-So400M features, attention-diversity loss, **384×384**
input.

| Result | Value |
|---|---|
| FF++ | 99.96% |
| Celeb-DF | 95.11% |
| DeeperForensics | 99.62% |
| UADFV | 97.01% |
| **FF++ → DeMamba** | **57.38 AUC** |
| **FF++ + GTA-V → DeMamba** | **93.75 AUC** |

*Why it matters:* validates the whole-frame choice and beats face-crop SOTA. But it trains
on FF++ vs FF++ + **GTA-V game footage** (not AI-generated video), and was **not** trained
on the GenVideo split — so the bidirectional regime decomposition is still open. Those last
two rows are the most relevant numbers in the literature to RQ1.

### VidAudit — Auditing Generalization in AI-Generated Video Detection ✅
Cakiroglu, Lu, Dalkilic & Kurban · arXiv:2606.31004, 2026 ·
Indiana University Bloomington + Hamad Bin Khalifa University

**Adopt this protocol; do not claim its controls as your own.**

- A three-feature **clip-length** classifier reaches LOGO AUC **0.998** on GenVidBench
  unaudited → **0.529 (chance)** under audit, measuring nothing about motion
- A 20-paper survey found **none** applying all six controls
- A CLIP baseline was **caught carrying dataset identity**
- At FPR 0.1%, multiple high-AUC methods fall to **single-digit recall** and the
  **leaderboard order changes**
- Recommends an audited **tuple**: AUC + above-floor margin + operating-point recall +
  calibration
- Releases 14 detectors behind one plugin API

*Six controls:* canonical codec re-encoding · clip-length/leakage filtering ·
real-vs-real dataset identity testing · matched retraining · multi-seed + bootstrap CIs ·
true cross-dataset evaluation.

### GenVideo / DeMamba ✅
Chen, Hong, Huang, Xu, Gu, Li, Lan, Zhu, Zhang, Wang & Li · arXiv:2405.19707 ·
also *Science China Information Sciences*, doi:10.1007/s11432-024-4894-0 ·
code: `github.com/chenhaoxing/DeMamba`

**The synthetic-regime training corpus.**

| Split | Real | Fake |
|---|---|---|
| Total | 1,223,511 | 1,078,838 |
| Train | 1,213,511 | 1,048,575 |
| OOD test | 10,000 | 8,588 |

Train reals from Kinetics-400 + Youku-mPLUG; **test reals from MSR-VTT** — a deliberate
domain shift. ⚠️ This does *not* solve source bias for you. Defines two protocols already:
cross-generator classification (generalizability) and degraded-video classification
(robustness). UNITE benchmarks against it, so using it makes you comparable.

### Omni-Fake 🟡
CVPR 2026 · *authors and full title unverified*

Formalises the ternary problem: **real / partially manipulated / fully synthetic**.
Video set ~100k real, 100k fully synthetic, 10k tampered; video OOD ~1k each.

*Why it matters:* removes "face-manipulated and fully synthetic should be treated as
distinct categories" from your novelty. **Verify this exists before writing related work
around it.**

---

## 2. Benchmark datasets — face manipulation

| Paper | Cite key | Key facts |
|---|---|---|
| **FaceForensics++** — Rössler et al., ICCV 2019 | `rossler2019faceforensics` | 1,000 reals × 4 manipulations (DF/F2F/FS/NT). Your TRAIN source. c23 throughout. |
| **Celeb-DF** — Li et al., CVPR 2020 | `li2020celebdf` | 590 Celeb-real + 300 YouTube-real = **890 real**; 5,639 fake. Official test list ~518 videos (178R/340F). TEST only. |
| **DeeperForensics-1.0** — Jiang et al., CVPR 2020 | `jiang2020deeperforensics` | 60k videos, 100 actors, **7 distortions × 5 levels**. ⚠️ **Ships no reals** — its real class is the 1,000 FF++ YouTube targets. Split by FF++ target ID first. |
| **DFDC** — Dolhansky et al., 2020 | `dolhansky2020dfdc` | 23,654 real / 104,500 fake. **Access blocked** (AWS-only gate). Cited, not used. |
| **DFDC Preview** — Dolhansky et al., 2019 | `dolhansky2019dfdcpreview` | 5,214 videos, 66 actors, 2 methods. Same gate. |
| **ForgeryNet** — He et al., CVPR 2021 (Oral) | `he2021forgerynet` | 221,247 videos, 15 approaches, 36 perturbations, 5,400+ subjects. **496 GB — dropped from core.** |
| **DF40** — Yan et al., NeurIPS 2024 | `yan2024df40` | 40 methods. ❌ Rejected: ships **face-cropped images**, incompatible with whole-frame. |
| **DeepFake-TIMIT** — Korshunov & Marcel, 2018 | `korshunov2018deepfaketimit` | 620 fakes, 64×64/128×128 faceswap-GAN. ❌ Rejected: obsolete, tiny, reals need VidTIMIT separately. |

## 3. Benchmark datasets — fully synthetic

| Paper | Cite key | Key facts |
|---|---|---|
| **DeepAction v1** — Bohacek & Farid, arXiv 2412.00526 | `bohacek2024deepaction` | **6.06 GB**, 2,600 videos, 6 T2V generators + Pexels real (100 videos). All **512×512** ⚠️ resolution shortcut risk. CC BY 4.0, no gate. **TEST only.** |
| **DeepSpeak v1.1/v2** — Barrington, Bohacek & Farid, arXiv 2408.05366 | `barrington2024deepspeak` | 500 participants, 100+ hrs, 14 video + 3 voice engines. Identity-matched. HF gated, ~1 month queue. Table 1 is the dataset-landscape survey. |
| **AIGVDBench** 🟡 | `aigvdbench` | 33 detectors, 31 generators, 440k+ videos. VidAudit's cross-dataset target. |
| **CoCoVideo-26K** 🟡 | `cocovideo26k` | 13 commercial generators, **semantically aligned real/fake pairs** — directly relevant to the source-matching rule. |
| **Corvi et al.** — generalizable AI-video detection, 2025 ⚠️ | `corvi2025seeing` | Forensic-oriented augmentation. Verify arXiv ID and venue. |

## 4. Cross-dataset generalization

| Paper | Cite key | Cross-dataset numbers worth knowing |
|---|---|---|
| **SBI** — Shiohara & Yamasaki, CVPR 2022 | `shiohara2022sbi` | **CDF 92.87 AUC** trained on FF++ c23. The strongest non-exotic cross-dataset result. Face-based — must justify why you can't use it. |
| **Face X-ray** — Li et al., CVPR 2020 | `li2020facexray` | Celeb-DF 74.20, DFDC 80.92 |
| **LipForensics** — Haliassos et al., CVPR 2021 | `haliassos2021lipforensics` | Celeb-DF 82.40, DFDC 73.50, DFo 97.60 |
| **RECCE** — Cao et al., CVPR 2022 | `cao2022recce` | Celeb-DF 68.71, DFDC 69.06 (from 99.32 in-dataset) |
| **Community Forensics** — Park & Owens, CVPR 2025 | `park2025community` | Thousands of generators for training. Closest to your unified-training question. |
| **F3-Net** — Qian et al., ECCV 2020 | `qian2020f3net` | Frequency-domain; FF++ 65.2 → Celeb-DF |
| **SPSL** — Liu et al., CVPR 2021 | `liu2021spsl` | Celeb-DF 76.9 |
| **Multi-Attentional** — Zhao et al., CVPR 2021 | `zhao2021multiattentional` | FF++ 99.29 in-dataset |

**The pattern to internalise:** in-dataset 96–99.9 AUC is near-saturated and means little.
Cross-dataset 65–85 is the real range. Only SBI-class methods break 90.

## 5. Shortcut learning / dataset bias

| Paper | Cite key | Why |
|---|---|---|
| **Shortcut Learning in Deep Neural Networks** — Geirhos et al., Nature MI 2020 | `geirhos2020shortcut` | The framework for your biggest risk. Read first. |
| **Unbiased Look at Dataset Bias** — Torralba & Efros, CVPR 2011 | `torralba2011unbiased` | The "name that dataset" experiment = your real-source probe, 14 years early. |
| **Recognition in Terra Incognita** — Beery et al., ECCV 2018 | `beery2018terra` | Classic background-shortcut demonstration. |

## 6. Temporal / video modelling

| Paper | Cite key | Note |
|---|---|---|
| **FTCN** — Zheng et al., ICCV 2021 | `zheng2021ftcn` | 32-frame clips, fully temporal |
| **MVFNet** — Nguyen & Stamm, WACV 2025 | `nguyen2025mvfnet` | Multiple forensic evidence forms |
| **X3D** — Feichtenhofer, CVPR 2020 | `feichtenhofer2020x3d` | ForgeryNet's best video baseline at **2.9M params** — argues against scaling |
| **SlowFast** — Feichtenhofer et al., ICCV 2019 | `feichtenhofer2019slowfast` | ForgeryNet: 97.28 AUC multi-crop |
| **TSM** — Lin et al., ICCV 2019 | `lin2019tsm` | Temporal shift |

*ForgeryNet's multi-crop finding:* +3.2 to +3.4 AUC from sampling more of the video, no
architecture change. Aggregation beats architecture.

## 7. Backbones

| Paper | Cite key | Use |
|---|---|---|
| **ConvNeXt** — Liu et al., CVPR 2022 | `liu2022convnext` | Arm 1 (supervised CNN) |
| **SigLIP** — Zhai et al., ICCV 2023 | `zhai2023siglip` | Arm 2 — UNITE uses SigLIP-So400M |
| **DINOv2** — Oquab et al., TMLR 2024 ⚠️ | `oquab2023dinov2` | Arm 2 alternative |
| **ViT** — Dosovitskiy et al., ICLR 2021 | `dosovitskiy2021vit` | cite only |
| **Swin** — Liu et al., ICCV 2021 | `liu2021swin` | cite only |
| **Xception** — Chollet, CVPR 2017 | `chollet2017xception` | the FF++ baseline detector |
| **EfficientNet** — Tan & Le, ICML 2019 | `tan2019efficientnet` | cite only |

## 8. Surveys — cite one or two, not six

`zhang2025unmasking` (JKSU-CIS) · `ali2025interframe` (Electronics)

## 9. Own artifacts

`george_dataset_hf` (Hugging Face datasets) · `george_code_github` (GitHub repo)

---

## Cut from the old manuscript — do not reinstate

~12 of the original 39 were irrelevant:

| Cut | Count | Reason |
|---|---|---|
| Medical imaging (segmentation, hyperspectral, medical image computing) | 6 | Zero connection to deepfake detection |
| Generic vision architectures (MambaVision, LAVIN-DiT, SpectFormer, HTR-ViT) | 4 | Not deepfake work |
| Unrelated forensics (cloud auth via facial forensics, copy-move forgery) | 2 | Different problem |
| Transformer surveys | several | One survey is enough |

A padded bibliography signals a weak literature review — and this manuscript already has
one rejection.

---

## Verification TODO before submission

1. **Confirm Omni-Fake exists** (authors, title, venue) — it's load-bearing for the novelty
   argument
2. **Confirm AIGVDBench and CoCoVideo-26K** — authors, venue, year
3. **Page numbers + DOIs** for every `[VERIFY]` entry — DBLP (`dblp.org`) has clean BibTeX
   for all CVPR/ICCV/ECCV/NeurIPS entries here. ~20 minutes.
4. **Corvi et al.** — arXiv ID and final venue
5. **DINOv2** — full author list and TMLR volume

Never submit an unverified page range.
