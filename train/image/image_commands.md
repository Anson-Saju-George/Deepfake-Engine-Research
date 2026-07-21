# Image Commands

This file is the command reference for the active image training tree under `train/image/`.

Experiment numbering is global and should be followed by `exp_no`, not by family grouping alone.

## Active Modules

- smoke test: `python -m train.image.simulate_image_train`
- image runners:
  - `python -m train.image.run_image`
  - `python -m train.image.run_image`
  - `python -m train.image.run_image`
  - `python -m train.image.run_image`
  - `python -m train.image.run_image`
  - `python -m train.image.run_image`
  - `python -m train.image.run_image`

## Shared Image Runner CLI

All active image runner modules expose the same parse surface:

Audit note:

- every active image runner now writes the full CLI transcript into `train.log` inside the resolved run directory
- this makes each completed run easier to cite and audit later

`--exp`

- type: string
- required for non-interactive use: yes
- behavior if omitted:
  - the runner opens an interactive experiment selection prompt

`--dataset-scope`

- type: string
- required for non-interactive use: yes
- behavior if omitted:
  - the runner opens an interactive dataset selection prompt
- allowed values:
  - `cifake`
  - `ai_gen`
  - `image_combined`

`--batch-size`

- type: integer
- required: no
- default behavior:
  - if omitted, the experiment registry batch size is used
- purpose:
  - temporary override for GPU memory or throughput tuning without changing registry defaults

## Dataset Scope Meanings

`cifake`

- runs only on `datasets/images/cifake`

`ai_gen`

- runs only on `datasets/images/ai-generated-images-vs-real-images`

`image_combined`

- runs on both image datasets together
- preserves per-source dataset boundaries during split preparation

## Smoke Test CLI

Module:

```bash
python -m train.image.simulate_image_train
```

Parse arguments:

`--datasets`

- type: one or more strings
- nargs: `+`
- default:
  - `cifake`
  - `ai-generated-images-vs-real-images`
- accepted values in current usage:
  - `cifake`
  - `ai-generated-images-vs-real-images`

`--batch-size`

- type: integer
- default: `64`

Examples:

```bash
python -m train.image.simulate_image_train
```

```bash
python -m train.image.simulate_image_train --datasets cifake
```

```bash
python -m train.image.simulate_image_train --datasets ai-generated-images-vs-real-images --batch-size 128
```

## Global Experiment Order

The current active image experiment ladder is:

- `IMG-EXP-01` -> `vit_base_patch16_224`
- `IMG-EXP-02` -> `vit_large_patch16_224`
- `IMG-EXP-03` -> `vit_huge_patch14_224`
- `IMG-EXP-04` -> `convnext_base`
- `IMG-EXP-05` -> `convnext_large`
- `IMG-EXP-06` -> `convnext_xlarge`
- `IMG-EXP-07` -> `swin_base_patch4_window7_224`
- `IMG-EXP-08` -> `swin_large_patch4_window7_224`
- `IMG-EXP-09` -> `deit3_base_patch16_224`
- `IMG-EXP-10` -> `convnextv2_base.fcmae_ft_in22k_in1k`
- `IMG-EXP-11` -> `convnextv2_large.fcmae_ft_in22k_in1k`
- `IMG-EXP-12` -> `maxvit_base_tf_224.in1k`
- `IMG-EXP-13` -> `eva02_base_patch14_224.mim_in22k_ft_in22k_in1k`
- `IMG-EXP-14` -> `eva02_large_patch14_224.mim_m38m_ft_in22k_in1k`

Current completed-result direction:

- image leader: `IMG-EXP-04` ConvNeXt-Base
- close second: `IMG-EXP-07` Swin-Base
- larger scale has not automatically helped, because `IMG-EXP-05` did not beat `IMG-EXP-04`

## ViT Runner

Module:

```bash
python -m train.image.run_image
```

Allowed `--exp` values:

- `IMG-EXP-01`
- `IMG-EXP-02`
- `IMG-EXP-03`

Examples:

```bash
python -m train.image.run_image --exp IMG-EXP-01 --dataset-scope cifake
```

```bash
python -m train.image.run_image --exp IMG-EXP-02 --dataset-scope image_combined --batch-size 16
```

Interactive mode:

```bash
python -m train.image.run_image
```

## ConvNeXt Runner

Module:

```bash
python -m train.image.run_image
```

Allowed `--exp` values:

- `IMG-EXP-04`
- `IMG-EXP-05`
- `IMG-EXP-06`

Examples:

```bash
python -m train.image.run_image --exp IMG-EXP-04 --dataset-scope image_combined --batch-size 32
```

```bash
python -m train.image.run_image --exp IMG-EXP-05 --dataset-scope image_combined --batch-size 16
```

```bash
python -m train.image.run_image --exp IMG-EXP-06 --dataset-scope image_combined --batch-size 8
```

## Swin Runner

Module:

```bash
python -m train.image.run_image
```

Allowed `--exp` values:

- `IMG-EXP-07`
- `IMG-EXP-08`

Examples:

```bash
python -m train.image.run_image --exp IMG-EXP-07 --dataset-scope image_combined --batch-size 16
```

```bash
python -m train.image.run_image --exp IMG-EXP-08 --dataset-scope image_combined --batch-size 8
```

## DeiT Runner

Module:

```bash
python -m train.image.run_image
```

Allowed `--exp` values:

- `IMG-EXP-09`

Examples:

```bash
python -m train.image.run_image --exp IMG-EXP-09 --dataset-scope image_combined --batch-size 16
```

## ConvNeXtV2 Runner

Module:

```bash
python -m train.image.run_image
```

Allowed `--exp` values:

- `IMG-EXP-10`
- `IMG-EXP-11`

Examples:

```bash
python -m train.image.run_image --exp IMG-EXP-10 --dataset-scope image_combined --batch-size 16
```

```bash
python -m train.image.run_image --exp IMG-EXP-11 --dataset-scope image_combined --batch-size 8
```

## MaxViT Runner

Module:

```bash
python -m train.image.run_image
```

Allowed `--exp` values:

- `IMG-EXP-12`

Examples:

```bash
python -m train.image.run_image --exp IMG-EXP-12 --dataset-scope image_combined --batch-size 8
```

## EVA Runner

Module:

```bash
python -m train.image.run_image
```

Allowed `--exp` values:

- `IMG-EXP-13`
- `IMG-EXP-14`

Examples:

```bash
python -m train.image.run_image --exp IMG-EXP-13 --dataset-scope image_combined --batch-size 16
```

```bash
python -m train.image.run_image --exp IMG-EXP-14 --dataset-scope image_combined --batch-size 4
```

## Save Layout

Image runs save under:

```text
train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/
```

## Current Runtime Note

The current shell environment is still missing `scikit-learn`, so the full image runners will fail at startup until that dependency is installed.
