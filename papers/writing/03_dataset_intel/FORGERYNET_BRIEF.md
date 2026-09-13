# ForgeryNet — dataset brief for DATASETS.md. Paste into Claude Code

Record this dataset fully. DO NOT download yet — record only, then await explicit go-ahead.

## IDENTITY
- **Name:** ForgeryNet (CVPR 2021 **Oral**)
- **Authors:** Yinan He, Bei Gan, Siyu Chen, Yichun Zhou, Guojun Yin, Luchuan Song,
  Lu Sheng, Jing Shao, Ziwei Liu
- **Official repo:** https://github.com/yinanhe/ForgeryNet
- **Project page:** https://yinanhe.github.io/projects/forgerynet.html
- **License:** RESTRICTED to non-commercial research and educational use
- **Role for us:** UNDECIDED — candidate TEST set (video). See "Concerns" before assigning.

## SCALE (from the authors' own comparison table)
| | Real | Fake |
|---|---|---|
| Video clips | 99,630 | 121,617 |
| Still images | 1,438,201 | 1,457,861 |

- **Total: 2.9M images + 221,247 videos**
- **15 manipulation approaches** (7 image-level + 8 video-level)
- **5,400+ subjects**
- **36 unique perturbations + mixed perturbations** (only DeeperForensics also has mixed)
- **9,393,574 annotations** (6.3M classification labels, 2.9M manipulated-area masks,
  221,247 temporal forgery segment labels)

Largest public face-forgery dataset by scale, manipulation count, subjects, and perturbations.

## CONFIRMED TOTAL SIZE: **496.23 GB**
(From a public share listing dated 2021-08-25. This is the first confirmed byte figure —
the official pages publish no GB number.)

## FOUR TASKS SUPPORTED (why the packages are split)
1. Image Forgery Classification (2-way / 3-way / n-way over 15 approaches)
2. Spatial Forgery Localization (manipulated-area segmentation)
3. **Video Forgery Classification** — note: manipulated frames are at RANDOM POSITIONS,
   deliberately, because real attackers manipulate arbitrary frames
4. Temporal Forgery Localization

## PACKAGE STRUCTURE
| Package | Contents | Needed for video-only work? |
|---|---|---|
| `Training.tar.*` (split parts) | train images + videos | only if training on it |
| `Validation.tar` | val split | optional |
| `public_test_images.tar` | image test set | ❌ image task only |
| **`public_test_videos.tar`** | **video test set** | ✅ **the minimal useful package** |
| `train_list.tar` / `val_list.tar` / `test_list.tar` | label files | ✅ required |

Training set ships as split archives: reassemble with `cat Training.tar.* >> Training.tar`.

## ⚠️ TWO DISTRIBUTIONS IN CIRCULATION — VERIFY WHICH YOU GET
Two different MD5 sets exist. They are NOT the same release:

**Set A (project page):**
```
272d385f3cac5dd6feabc25711c18a41  Training.tar
df320faca2128e022764f8d0e4bc75a1  Validation.tar
e4218faa4b934345977202edfbf89301  public_test_images.tar
92b870009cb03832b2c2795b6a35629f  public_test_videos.tar
```

**Set B (README, different naming — note `_with_real`):**
```
750e34521e61f81486a678d5b76a5ef3  Training.tar
c6fd18252b5e0ae04a5b163297b5cb28  Validation_with_real.tar
7d2c0cf50bdc77890ace1ba4461e1893  Test_with_real.tar
8558b99366ccb93cd9da56f600f0901f  test_list.tar
8bda9b0a6dc7721e5f1d0b6b3793ce2a  train_list.tar
a280bb501166a01b8d376df2b5b9ae13  val_list.tar
```
**Record BOTH. On download, MD5 every archive and note which set it matches — the label
files must match the data release or annotations will be misaligned.**

## DIRECTORY LAYOUT (from official README)
```
Train/
  images/           15 "fake" subdirs (one per method) + 4 "real" subdirs
  videos/            8 "fake" subdirs (one per method) + 4 "real" subdirs
  spatial_localize/  per-image forgery location masks
```
**Image list format:** `<image_name> <binary_cls_label> <triple_cls_label> <16cls_label>`
**Video list format:** `<segmentation length> <video_name> [<start frame> <during frames count> <fake/real>]*N <binary_cls_label> <triple_cls_label> <16cls_label>`

Note the video labels carry per-segment temporal annotations — usable for binary
classification by taking `<binary_cls_label>` only.

## MANIPULATION TAXONOMY
- **(a) ID-remained:** Talking-head Video, FirstOrder Motion, ATVG-Net, MaskGAN,
  SC-FEGAN, StarGAN2, StyleGAN2, DiscoFaceGAN
- **(b) ID-replaced:** BlendFace, MM Replacement, DeepFakes, FSGAN, FaceShifter, SBS, DSS

## PACKAGE / BUTTON INVENTORY — KNOW ALL OF THESE, BUT DO NOT TAKE ALL OF THEM

Complete list of what the official project page offers:

| Button on project page | Package | Contents | Take it? |
|---|---|---|---|
| DOWNLOAD TRAINING SET | `Training.tar.*` (split parts) | train **images + videos** | ❌ covered by TRAIN&VAL |
| DOWNLOAD VALIDATION SET | `Validation.tar` | val **images + videos** | ❌ covered by TRAIN&VAL |
| **DOWNLOAD TRAIN&VAL** | combined archive | **Training + Validation together** | ✅ **TAKE** |
| DOWNLOAD IMAGE TEST SET | `public_test_images.tar` | test split, **images only** | ✅ **TAKE** |
| DOWNLOAD VIDEO TEST SET | `public_test_videos.tar` | test split, **videos only** | ✅ **TAKE** |
| DOWNLOAD TRAINING LIST | `train_list.tar` | train labels | ✅ **TAKE** |
| DOWNLOAD VALIDATION LIST | `val_list.tar` | val labels | ✅ **TAKE** |
| (listed in MD5 set B) | `test_list.tar` | test labels | ✅ **TAKE if present** |
| README | docs | directory/label spec | ✅ **TAKE** |

### ⚠️ THE DUPLICATION TRAP — do not download both paths
`DOWNLOAD TRAIN&VAL` is an ALTERNATIVE to downloading `Training` + `Validation`
separately. Taking both = downloading the same data twice (~hundreds of GB wasted).
**Use TRAIN&VAL only.**

### Why no separate image/video download for train or val
Per the official README, `Training.tar` contains BOTH `images/` and `videos/`
subdirectories. Same for validation. So train and val already include both modalities —
there is nothing extra to fetch for them.

The **test split is the only place images and videos ship as separate archives**. That is
NOT duplication — it is the same split cut by task, so both are required for complete
test coverage.

## ✅ EXACT DOWNLOAD LIST (this and nothing else)
1. **TRAIN&VAL** (combined) — train + val, images + videos
2. **`public_test_images.tar`** — test, image task
3. **`public_test_videos.tar`** — test, video task
4. **`train_list.tar`** — train labels
5. **`val_list.tar`** — val labels
6. **`test_list.tar`** — test labels (if the served release includes it)
7. **README**

**DO NOT take:** `DOWNLOAD TRAINING SET` and `DOWNLOAD VALIDATION SET` as separate
downloads — they are already inside TRAIN&VAL.

⚠️ **`test_list.tar` appears only in MD5 set B** (the `_with_real` release); the
project-page set does not list it. The two releases package labels differently. If the
served release uses `Validation_with_real.tar` / `Test_with_real.tar` naming, the package
names will NOT match the button names above — map them by content, MD5-verify, and record
which set you received.

## DOWNLOAD ROUTES (ranked by trustworthiness) — USE ROUTE 1

### ⭐ ROUTE 1 — OpenDataLab (PREFERRED — use this)
**https://opendatalab.com/OpenDataLab/ForgeryNet**
- Listed size: **499.5 GB** · 35k downloads · 46.3k / 4.6k engagement
- Legitimate academic data platform. Solves the Aliyun-Drive access wall reported in
  repo issue #11 (Chinese UI, account required, a researcher reported being unable to
  download).
- **Has a CLI/SDK** — this is the deciding factor for a 500 GB pull: resumable and
  scriptable. A browser download of this size WILL fail partway; use the CLI.
- Requires an OpenDataLab account. Check their CLI docs on the dataset page
  ("CLI/SDK download" tab) for the exact command and auth setup.

### ROUTE 2 — Official project page (Google Drive buttons)
**https://yinanhe.github.io/projects/forgerynet.html**
Buttons: DOWNLOAD TRAINING SET / VALIDATION SET / IMAGE TEST SET / VIDEO TEST SET /
TRAINING LIST / VALIDATION LIST / README / DOWNLOAD TRAIN&VAL
- Authoritative source, but Google Drive for ~500 GB is fragile (quota limits, no
  resume). Training set ships as split parts needing `cat Training.tar.* >> Training.tar`.
- Repo: https://github.com/yinanhe/ForgeryNet

### ⚠️ ROUTE 3 — third-party re-shares (LAST RESORT ONLY)
Provenance unverifiable; MD5s may match NEITHER official set. Do not use unless
Routes 1 and 2 both fail, and MD5-verify every archive before use.
Recorded for completeness only:
- Bare Google Drive file IDs (no filenames given, contents unverified):
  - https://drive.google.com/file/d/1conYQXWguAwJ1eEwewHyMBGUtgjgR_sM/view
  - https://drive.google.com/file/d/1CSJOkDR_jJvq7qUGP8oGcwpoPdKGzfgb/view
  - https://drive.google.com/file/d/1ZqPqmYdqmq4_HLZu1c5ySXhcR1WQWQ9L/view
  - https://drive.google.com/file/d/14Rqq4C4oHK6FEqyTIiVcQy7wQuMHGQzu/view
- 115cdn password-protected share (password `cvpr`), listing "ForgeryNet 495.23 GB,
  modified 2021-08-25": https://115cdn.com/s/swnk84d3wl3

### Note on the size discrepancy
OpenDataLab reports 499.5 GB; the 115cdn share reports 495.5 GB. A ~0.8% (~4 GB) gap is
GB-vs-GiB unit reporting and/or whether the small label archives are counted — NOT a
different dataset version. A genuinely different release would differ by far more.
**The MD5s, not the byte count, determine which release you have.** Verify on arrival.

**Known infrastructure problems:** train/val were moved to Aliyun Drive (Chinese, no
English UI, account required); repo issue #11 reports a researcher unable to download.
Repo has ~12+ open issues since 2023, several unanswered (latest #52, Nov 2025). Expect
no maintainer support.

## CONCERNS TO RECORD (do not omit — these decide the paper role)
1. **Comparability:** essentially nobody trains on ForgeryNet, so cross-dataset numbers
   trained on it cannot be placed against published FF++-trained tables.
2. **Image and video subsets contain DIFFERENT subjects** — this cuts against a
   protocol-separated image-vs-video comparison, which is this project's thesis.
3. **Long-tailed manipulation distribution** in the training set — introduces a new
   imbalance problem.
4. **496 GB + days of frame extraction**; images are an additional large cost on top.
5. Video task deliberately places manipulated frames at random positions — a fixed
   centre-clip sampler would systematically miss them.

## USER'S POSITION (record as-is)
User has ~1.2 TB free and accepts the 496 GB cost. Wants BOTH image and video sets
downloaded so the image track can be explored later if desired. Role in the paper
(train vs test, video-only vs image+video) remains UNDECIDED.

## ACTION
Record ALL of the above in `DATASETS.md` — the full package inventory, both MD5 sets, all
three routes, and the concerns. Future-you should be able to answer any question about
this dataset from that record.

**But when downloading, take ONLY the 7 items in the EXACT DOWNLOAD LIST above.** Knowing
about every package is not permission to fetch every package — the duplication trap
(TRAIN&VAL vs separate Training+Validation) would waste hundreds of GB.

**Download nothing yet.** Await explicit go-ahead specifying: (a) confirm the 7-item list,
(b) which route (default: OpenDataLab CLI), (c) target drive.

**When the go-ahead comes, suggested order** (validate the chain before committing to the
bulk): pull the small items first — README, the three `*_list.tar` label archives, and
`public_test_videos.tar`. MD5-verify them, confirm which release set they match, confirm
the label format parses and the directory layout matches the README. ONLY THEN start the
large TRAIN&VAL pull. This catches a wrong-release or label-mismatch problem in an hour
instead of after a multi-day download.
