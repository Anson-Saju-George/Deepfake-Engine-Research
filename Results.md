# Results

Current held-out test results derived from the saved `test_evaluation.json` files under `train/`.

Artifacts used:

- prediction exports: `test_predictions.csv`
- evaluation summaries: `test_evaluation.json`
- graph index: `graphs/graph_manifest.csv`

## Best Overall

- Best image run: `IMG-EXP-04 | ConvNeXt | convnext_base`
  - accuracy: `0.9863`
  - f1: `0.9863`
  - precision: `0.9867`
  - recall: `0.9858`
  - roc_auc: `0.9968`
  - average_precision: `0.9956`
- Best video run: `VID-TMP-02 | ConvNeXt Sequence | convnext_large`
  - accuracy: `0.9089`
  - f1: `0.7841`
  - precision: `0.7105`
  - recall: `0.8747`
  - roc_auc: `0.9594`
  - average_precision: `0.8924`
- Matching video result: `VID-ST-03 | ConvNeXt Hybrid | convnext_large`
  - accuracy: `0.9089`
  - f1: `0.7841`
  - precision: `0.7105`
  - recall: `0.8747`
  - roc_auc: `0.9594`
  - average_precision: `0.8924`

## Image Results

| Experiment | Family | Model | Accuracy | F1 | Precision | Recall | ROC-AUC | AP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IMG-EXP-04 | ConvNeXt | convnext_base | 0.9863 | 0.9863 | 0.9867 | 0.9858 | 0.9968 | 0.9956 |
| IMG-EXP-07 | Swin | swin_base_patch4_window7_224 | 0.9842 | 0.9842 | 0.9860 | 0.9823 | 0.9987 | 0.9986 |
| IMG-EXP-05 | ConvNeXt | convnext_large | 0.9840 | 0.9840 | 0.9831 | 0.9849 | 0.9966 | 0.9966 |
| IMG-EXP-01 | ViT | vit_base_patch16_224 | 0.9703 | 0.9702 | 0.9734 | 0.9670 | 0.9939 | 0.9931 |
| IMG-EXP-02 | ViT | vit_large_patch16_224 | 0.9548 | 0.9546 | 0.9588 | 0.9504 | 0.9899 | 0.9893 |

## Video Results

| Experiment | Category | Family | Model | Loss | Accuracy | F1 | Precision | Recall | ROC-AUC | AP |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VID-TMP-02 | temporal | ConvNeXt Sequence | convnext_large | none | 0.9089 | 0.7841 | 0.7105 | 0.8747 | 0.9594 | 0.8924 |
| VID-ST-03 | spatiotemporal | ConvNeXt Hybrid | convnext_large | none | 0.9089 | 0.7841 | 0.7105 | 0.8747 | 0.9594 | 0.8924 |
| VID-TMP-01 | temporal | ConvNeXt Sequence | convnext_base | weighted_ce | 0.8828 | 0.7092 | 0.6679 | 0.7559 | 0.9217 | 0.8047 |
| VID-TMP-01 | temporal | ConvNeXt Sequence | convnext_base | none | 0.8652 | 0.7090 | 0.5991 | 0.8683 | 0.9414 | 0.8445 |
| VID-SPA-02 | spatial | ConvNeXt | convnext_base | none | 0.8566 | 0.7023 | 0.5782 | 0.8942 | 0.9498 | 0.8682 |
| VID-SPA-02 | spatial | ConvNeXt | convnext_base | weighted_ce | 0.8717 | 0.6709 | 0.6517 | 0.6911 | 0.8941 | 0.7443 |
| VID-ST-05 | spatiotemporal | MaxViT Hybrid | maxvit_base_tf_224.in1k | none | 0.8186 | 0.6414 | 0.5123 | 0.8575 | 0.9161 | 0.7482 |
| VID-ST-02 | spatiotemporal | ConvNeXt Hybrid | convnext_base | none | 0.7663 | 0.5744 | 0.4381 | 0.8337 | 0.8586 | 0.6036 |
| VID-SPA-06 | spatial | Swin | swin_base_patch4_window7_224 | none | 0.7504 | 0.5253 | 0.4102 | 0.7300 | 0.8145 | 0.5507 |
| VID-SPA-02 | spatial | ConvNeXt | convnext_base | focal | 0.8027 | 0.4744 | 0.4781 | 0.4708 | 0.7059 | 0.4398 |

## Category Winners

- Image: `IMG-EXP-04 | ConvNeXt | convnext_base`
- Spatial video: `VID-SPA-02 | ConvNeXt | convnext_base | loss=none`
- Temporal video: `VID-TMP-02 | ConvNeXt Sequence | convnext_large | loss=none`
- Spatiotemporal video: `VID-ST-03 | ConvNeXt Hybrid | convnext_large | loss=none`

## Current Takeaways

- Image detection is much stronger than the current video runs in this repo snapshot.
- `ConvNeXt` is the strongest family overall.
- For video, `ConvNeXt-Large` dominates the completed sequence and hybrid runs.
- `focal` loss underperformed on the completed spatial ConvNeXt comparison.
- `VID-TMP-02` and `VID-ST-03` are numerically identical in the current evidence snapshot. This matches the implementation note in `train/video/video_config.md` that the active temporal and spatiotemporal branches currently share the same sequence trainer mechanics.

## Reproduction

Export prediction-based evaluation:

```powershell
python -m train.image.test_image_models --workers 8 --prefetch-factor 4 --batch-size 128
python -m train.video.test_video_models --workers 8 --prefetch-factor 4 --batch-size 4
```

Regenerate graphs:

```powershell
python graphs\generate_research_graphs.py
```
