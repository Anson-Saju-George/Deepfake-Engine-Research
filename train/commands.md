# Training Commands

This file is the training-side command sheet for the current repo state.

Use this file for:

- training entrypoints
- smoke-test entrypoints
- CLI argument reference
- implemented experiment IDs and dataset-scope tags

Do not use this file as the dataset/pipeline audit trail. That remains in:

- `data/dataset.md`
- `data/commands.md`

## Current Training State

- active implemented image training tree: `train/`
- current implemented image smoke-test module: `python -m train.image_simulate`
- current implemented image trainer module: `python -m train.run_image_vit`
- current implemented image trainer family: `ViT`
- current migrated video execution in `train/`: not implemented yet
- video planning/registry documents exist:
  - `train/video_model.py`
  - `train/video_config.md`

## Save Layout

Current implemented image save layout:

```text
train/image/<family_name>/<exp_no>_<model_name>_<dataset_tag>/
```

Current family directory names:

- `ViT`
- `ConvNeXt`
- `EfficientNet`
- `ResNet`

Current dataset tags:

- `cifake`
- `ai_gen`
- `image_combined`

## Console Progress Behavior

Current implemented training console behavior:

- each phase uses its own tqdm
- train, validation, and test progress bars stay visible after completion
- phase labels remain fixed
- live loss is shown in tqdm postfix rather than overwriting the phase label

Current phase labels used by the implemented image trainer:

- `train e<epoch>`
- `val e<epoch>`
- `test`

## Image Smoke Test

Module:

```bash
python -m train.image_simulate
```

Purpose:

- validate the image-only dataloader path end to end
- run 1 train epoch
- run validation and test passes
- catch split, decode, or loader issues before expensive training

### Default behavior

```bash
python -m train.image_simulate
```

Current default datasets:

- `cifake`
- `ai-generated-images-vs-real-images`

### Single-dataset examples

```bash
python -m train.image_simulate --datasets cifake
```

```bash
python -m train.image_simulate --datasets ai-generated-images-vs-real-images
```

### Batch-size override

```bash
python -m train.image_simulate --batch-size 128
```

### All currently usable parse variables

`--datasets`

- type: one or more strings
- nargs: `+`
- default:
  - `cifake`
  - `ai-generated-images-vs-real-images`
- currently usable values:
  - `cifake`
  - `ai-generated-images-vs-real-images`

`--batch-size`

- type: integer
- default: `64`
- example values:
  - `32`
  - `64`
  - `128`

## ViT Image Trainer

Module:

```bash
python -m train.run_image_vit
```

Purpose:

- run the currently implemented ViT image trainer
- uses the active `image_only` dataloader path
- supports ViT experiments `IMG-EXP-01..03`

### Interactive menu mode

If no CLI selection arguments are passed, the script opens a simple menu:

```bash
python -m train.run_image_vit
```

The menu will ask for:

- experiment number
- dataset scope

### Direct CLI examples

Run ViT-Base on CIFAKE:

```bash
python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope cifake
```

Run ViT-Base on AI-generated-images-vs-real-images:

```bash
python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope ai_gen
```

Run ViT-Base on combined image datasets:

```bash
python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope image_combined
```

Run ViT-Large on combined image datasets:

```bash
python -m train.run_image_vit --exp IMG-EXP-02 --dataset-scope image_combined
```

Run ViT-Huge on combined image datasets:

```bash
python -m train.run_image_vit --exp IMG-EXP-03 --dataset-scope image_combined
```

Override batch size:

```bash
python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope cifake --batch-size 128
```

### All currently usable parse variables

`--exp`

- type: string
- default: none
- behavior if omitted:
  - interactive menu selection
- currently usable values:
  - `IMG-EXP-01`
  - `IMG-EXP-02`
  - `IMG-EXP-03`

Experiment mapping:

- `IMG-EXP-01` -> `vit_base_patch16_224`
- `IMG-EXP-02` -> `vit_large_patch16_224`
- `IMG-EXP-03` -> `vit_huge_patch14_224`

`--dataset-scope`

- type: string
- default: none
- behavior if omitted:
  - interactive menu selection
- currently usable values:
  - `cifake`
  - `ai_gen`
  - `image_combined`

Dataset-scope mapping:

- `cifake` -> `['cifake']`
- `ai_gen` -> `['ai-generated-images-vs-real-images']`
- `image_combined` -> `['cifake', 'ai-generated-images-vs-real-images']`

`--batch-size`

- type: integer
- default: no CLI override
- actual run default if omitted:
  - inherited from registry: `64`
- example values:
  - `32`
  - `64`
  - `128`

## Current ViT Trainer Behavior

The current implemented trainer in `train/image_train_vit.py` uses:

- protocol: `image_only`
- epochs: `10`
- batch size: `64` unless overridden
- balanced sampling: `True`
- optimizer: `AdamW`
- scheduler: cosine annealing
- label smoothing: `0.1`
- EMA: enabled
- checkpoint threshold for promoted final export: validation accuracy `>= 0.80`

## Current Output Files Per Implemented Image Run

Every completed image run currently writes:

- `best.pth`
- `last.pth`
- `config.json`
- `history.csv`
- `best_summary.json`
- `final_summary.json`
- `split_summary.json`
- `run_record.md`

Optional promoted final checkpoint:

- `final_<exp_no>_<model_name>_<dataset_tag>_valf1..._valacc....pth`

## Current Gaps

- ConvNeXt trainer in the new tree: not implemented yet
- EfficientNet trainer in the new tree: not implemented yet
- ResNet trainer in the new tree: not implemented yet
- new-tree video execution runner: not implemented yet

## Recommendation Before Real Training

Run this first:

```bash
python -m train.image_simulate
```

Then launch the actual image trainer, for example:

```bash
python -m train.run_image_vit --exp IMG-EXP-01 --dataset-scope image_combined
```
