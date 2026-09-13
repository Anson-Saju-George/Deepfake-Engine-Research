# Revision Audit for Completed Deepfake Experiments

Generated from existing repository artifacts only. No model was retrained and no training entrypoint was executed.

Audit root (at the time this audit was generated): `D:\main-projects\DEEPFAKE_MODEL-DS-APP\deep-fake-model`

Note: the repo was relocated after this audit was written and now lives at
`D:\main-projects\LIVE-ACTIVE\DEEPFAKE_MODEL-DS-APP\deep-fake-model`. The path above is kept as a
historical record of where this specific audit ran, not a current filesystem reference.

Primary artifact criteria:

- completed run = folder under `train/image` or `train/video` containing `final_summary.json`
- detailed test metrics = `test_evaluation.json` when present
- configuration = `config.json`
- split counts = `split_summary.json`
- epoch count = number of completed rows in `history.csv`

Important notation:

- `NA` means the requested value was not present in completed artifacts.
- Confusion matrix abbreviations: `RR=actual_real_pred_real`, `RF=actual_real_pred_fake`, `FR=actual_fake_pred_real`, `FF=actual_fake_pred_fake`.
- For newer temporal-head runs, `final_summary.json` exists but `test_evaluation.json` was not found, so precision, recall, ROC-AUC, average precision, and confusion matrix are not available from completed artifacts.

## Completed Artifact Inventory

Completed runs found:

- image: `5`
- video: `17`
- total: `22`

| ID | Family | Architecture | Category | Dataset scope | Completed folder |
|---|---|---|---|---|---|
| IMG-EXP-01 | ViT | `vit_base_patch16_224` | image | `image_combined` | `train/image/ViT/IMG-EXP-01_vit_base_patch16_224_image_combined` |
| IMG-EXP-02 | ViT | `vit_large_patch16_224` | image | `image_combined` | `train/image/ViT/IMG-EXP-02_vit_large_patch16_224_image_combined` |
| IMG-EXP-04 | ConvNeXt | `convnext_base` | image | `image_combined` | `train/image/ConvNeXt/IMG-EXP-04_convnext_base_image_combined` |
| IMG-EXP-05 | ConvNeXt | `convnext_large` | image | `image_combined` | `train/image/ConvNeXt/IMG-EXP-05_convnext_large_image_combined` |
| IMG-EXP-07 | Swin | `swin_base_patch4_window7_224` | image | `image_combined` | `train/image/Swin/IMG-EXP-07_swin_base_patch4_window7_224_image_combined` |
| VID-SPA-02 | ConvNeXt | `convnext_base` | spatial video | `video_combined` | `train/video/spa/ConvNeXt/VID-SPA-02_convnext_base_video_combined_loss-none` |
| VID-SPA-02 | ConvNeXt | `convnext_base` | spatial video | `video_combined` | `train/video/spa/ConvNeXt/VID-SPA-02_convnext_base_video_combined_loss-weighted_ce` |
| VID-SPA-02 | ConvNeXt | `convnext_base` | spatial video | `video_combined` | `train/video/spa/ConvNeXt/VID-SPA-02_convnext_base_video_combined_loss-focal` |
| VID-SPA-06 | Swin | `swin_base_patch4_window7_224` | spatial video | `video_combined` | `train/video/spa/Swin/VID-SPA-06_swin_base_patch4_window7_224_video_combined_loss-none` |
| VID-TMP-01 | ConvNeXt Sequence | `convnext_base` | temporal video | `video_combined` | `train/video/tmp/ConvNeXt_Sequence/VID-TMP-01_convnext_base_video_combined_loss-none` |
| VID-TMP-01 | ConvNeXt Sequence | `convnext_base` | temporal video | `video_combined` | `train/video/tmp/ConvNeXt_Sequence/VID-TMP-01_convnext_base_video_combined_loss-weighted_ce` |
| VID-TMP-02 | ConvNeXt Sequence | `convnext_large` | temporal video | `video_combined` | `train/video/tmp/ConvNeXt_Sequence/VID-TMP-02_convnext_large_video_combined_loss-none_lr-5e-05` |
| VID-TMP-07 | ConvNeXt LSTM | `convnext_large` | temporal video | `video_all` | `train/video/tmp/ConvNeXt_LSTM/VID-TMP-07_convnext_large_video_all_loss-none_lr-5e-05` |
| VID-TMP-07 | ConvNeXt LSTM | `convnext_large` | temporal video | `video_combined` | `train/video/tmp/ConvNeXt_LSTM/VID-TMP-07_convnext_large_video_combined_loss-none_lr-5e-05` |
| VID-TMP-08 | ConvNeXt Temporal Transformer | `convnext_large` | temporal video | `video_all` | `train/video/tmp/ConvNeXt_Temporal_Transformer/VID-TMP-08_convnext_large_video_all_loss-none_lr-5e-05` |
| VID-TMP-09 | ConvNeXt TCN | `convnext_large` | temporal video | `video_all` | `train/video/tmp/ConvNeXt_TCN/VID-TMP-09_convnext_large_video_all_loss-none_lr-5e-05` |
| VID-ST-02 | ConvNeXt Hybrid | `convnext_base` | spatiotemporal video | `video_combined` | `train/video/st/ConvNeXt_Hybrid/VID-ST-02_convnext_base_video_combined_loss-none` |
| VID-ST-03 | ConvNeXt Hybrid | `convnext_large` | spatiotemporal video | `video_combined` | `train/video/st/ConvNeXt_Hybrid/VID-ST-03_convnext_large_video_combined_loss-none_lr-5e-05` |
| VID-ST-05 | MaxViT Hybrid | `maxvit_base_tf_224.in1k` | spatiotemporal video | `video_combined` | `train/video/st/MaxViT_Hybrid/VID-ST-05_maxvit_base_tf_224.in1k_video_combined_loss-none_lr-5e-05` |
| VID-ST-07 | ConvNeXt ConvLSTM | `convnext_large` | spatiotemporal video | `video_all` | `train/video/st/ConvNeXt_ConvLSTM/VID-ST-07_convnext_large_video_all_loss-none_lr-5e-05` |
| VID-ST-08 | ConvNeXt Hybrid Transformer | `convnext_large` | spatiotemporal video | `video_all` | `train/video/st/ConvNeXt_Hybrid_Transformer/VID-ST-08_convnext_large_video_all_loss-none_lr-5e-05` |
| VID-ST-09 | ConvNeXt Hybrid TCN | `convnext_large` | spatiotemporal video | `video_all` | `train/video/st/ConvNeXt_Hybrid_TCN/VID-ST-09_convnext_large_video_all_loss-none_lr-5e-05` |

## Main Model Table

Requested main set: ConvNeXt, Swin, ViT for image; ConvNeXt spatial, ConvNeXt sequence, ConvNeXt-TCN, ConvNeXt-LSTM for video.

| ID | Family | Architecture | Category | Scope | Datasets | Protocol | Mode | Seq | Batch | Epochs run | LR | Loss | Train/Val/Test | Acc | F1 | Precision | Recall | ROC-AUC | AP | CM | Best epoch |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| IMG-EXP-04 | ConvNeXt | `convnext_base` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 32 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9863 | 0.9863 | 0.9867 | 0.9858 | 0.9968 | 0.9956 | RR=15770, RF=227, FR=212, FF=15788 | 8 |
| IMG-EXP-05 | ConvNeXt | `convnext_large` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 16 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9840 | 0.9840 | 0.9831 | 0.9849 | 0.9966 | 0.9966 | RR=15755, RF=242, FR=271, FF=15729 | 9 |
| IMG-EXP-07 | Swin | `swin_base_patch4_window7_224` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 32 | 8 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9842 | 0.9842 | 0.9860 | 0.9823 | 0.9987 | 0.9986 | RR=15714, RF=283, FR=223, FF=15777 | 5 |
| IMG-EXP-01 | ViT | `vit_base_patch16_224` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 64 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9703 | 0.9702 | 0.9734 | 0.9670 | 0.9939 | 0.9931 | RR=15469, RF=528, FR=423, FF=15577 | 9 |
| IMG-EXP-02 | ViT | `vit_large_patch16_224` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 16 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9548 | 0.9546 | 0.9588 | 0.9504 | 0.9899 | 0.9893 | RR=15203, RF=794, FR=653, FF=15347 | 10 |
| VID-SPA-02 | ConvNeXt | `convnext_base` | spatial video | video_combined | celeb-df-v2, faceforensics++ | video_only | single | 1 | 8 | 9 | 0.0001 | none | 14096/866/2448 | 0.8566 | 0.7023 | 0.5782 | 0.8942 | 0.9498 | 0.8682 | RR=414, RF=49, FR=302, FF=1683 | 6 |
| VID-TMP-02 | ConvNeXt Sequence | `convnext_large` | temporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14096/866/2448 | 0.9089 | 0.7841 | 0.7105 | 0.8747 | 0.9594 | 0.8924 | RR=405, RF=58, FR=165, FF=1820 | 10 |
| VID-TMP-09 | ConvNeXt TCN | `convnext_large` | temporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14140/872/2462 | 0.9102 | 0.7831 | NA | NA | NA | NA | NA | 10 |
| VID-TMP-07 | ConvNeXt LSTM | `convnext_large` | temporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 7 | 5e-05 | none | 14140/872/2462 | 0.8794 | 0.7336 | NA | NA | NA | NA | NA | 4 |
| VID-TMP-07 | ConvNeXt LSTM | `convnext_large` | temporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 4 | 2 | 9 | 5e-05 | none | 14096/866/2448 | 0.8779 | 0.7274 | NA | NA | NA | NA | NA | 6 |

## Secondary Ablation Table

Requested secondary set: ConvLSTM, Temporal Transformer, Hybrid Transformer, Hybrid TCN, MaxViT Hybrid, focal loss, weighted CE.

| ID | Ablation | Architecture | Category | Scope | Loss | Seq | Batch | Epochs run | Train/Val/Test | Acc | F1 | Precision | Recall | ROC-AUC | AP | CM | Best epoch |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| VID-ST-07 | ConvLSTM | `convnext_large` | spatiotemporal video | video_all | none | 4 | 2 | 10 | 14140/872/2462 | 0.8786 | 0.7342 | NA | NA | NA | NA | NA | 10 |
| VID-TMP-08 | Temporal Transformer | `convnext_large` | temporal video | video_all | none | 4 | 2 | 7 | 14140/872/2462 | 0.8733 | 0.7249 | NA | NA | NA | NA | NA | 4 |
| VID-ST-08 | Hybrid Transformer | `convnext_large` | spatiotemporal video | video_all | none | 4 | 2 | 10 | 14140/872/2462 | 0.8956 | 0.7591 | NA | NA | NA | NA | NA | 9 |
| VID-ST-09 | Hybrid TCN | `convnext_large` | spatiotemporal video | video_all | none | 4 | 2 | 10 | 14140/872/2462 | 0.8964 | 0.7632 | NA | NA | NA | NA | NA | 7 |
| VID-ST-05 | MaxViT Hybrid | `maxvit_base_tf_224.in1k` | spatiotemporal video | video_combined | none | 4 | 2 | 5 | 14096/866/2448 | 0.8186 | 0.6414 | 0.5123 | 0.8575 | 0.9161 | 0.7482 | RR=397, RF=66, FR=378, FF=1607 | 2 |
| VID-SPA-02 | Focal loss | `convnext_base` | spatial video | video_combined | focal | 1 | 4 | 5 | 8648/866/2448 | 0.8027 | 0.4744 | 0.4781 | 0.4708 | 0.7059 | 0.4398 | RR=218, RF=245, FR=238, FF=1747 | 2 |
| VID-SPA-02 | Weighted CE | `convnext_base` | spatial video | video_combined | weighted_ce | 1 | 8 | 6 | 8648/866/2448 | 0.8717 | 0.6709 | 0.6517 | 0.6911 | 0.8941 | 0.7443 | RR=320, RF=143, FR=171, FF=1814 | 3 |
| VID-TMP-01 | Weighted CE | `convnext_base` | temporal video | video_combined | weighted_ce | 8 | 2 | 10 | 8648/866/2448 | 0.8828 | 0.7092 | 0.6679 | 0.7559 | 0.9217 | 0.8047 | RR=350, RF=113, FR=174, FF=1811 | 7 |

## All Completed Runs: Metrics and Artifact Fields

| ID | Family | Architecture | Category | Scope | Datasets | Protocol | Mode | Seq | Batch | Epochs run | LR | Loss | Train/Val/Test | Acc | F1 | Precision | Recall | ROC-AUC | AP | CM | Best epoch |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| IMG-EXP-01 | ViT | `vit_base_patch16_224` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 64 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9703 | 0.9702 | 0.9734 | 0.9670 | 0.9939 | 0.9931 | RR=15469, RF=528, FR=423, FF=15577 | 9 |
| IMG-EXP-02 | ViT | `vit_large_patch16_224` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 16 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9548 | 0.9546 | 0.9588 | 0.9504 | 0.9899 | 0.9893 | RR=15203, RF=794, FR=653, FF=15347 | 10 |
| IMG-EXP-04 | ConvNeXt | `convnext_base` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 32 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9863 | 0.9863 | 0.9867 | 0.9858 | 0.9968 | 0.9956 | RR=15770, RF=227, FR=212, FF=15788 | 8 |
| IMG-EXP-05 | ConvNeXt | `convnext_large` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 16 | 10 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9840 | 0.9840 | 0.9831 | 0.9849 | 0.9966 | 0.9966 | RR=15755, RF=242, FR=271, FF=15729 | 9 |
| IMG-EXP-07 | Swin | `swin_base_patch4_window7_224` | image | image_combined | cifake, ai-generated-images-vs-real-images | image_only | single | NA | 32 | 8 | 0.0001 | cross_entropy | 126094/22199/31997 | 0.9842 | 0.9842 | 0.9860 | 0.9823 | 0.9987 | 0.9986 | RR=15714, RF=283, FR=223, FF=15777 | 5 |
| VID-SPA-02 | ConvNeXt | `convnext_base` | spatial video | video_combined | celeb-df-v2, faceforensics++ | video_only | single | 1 | 4 | 5 | 0.0001 | focal | 8648/866/2448 | 0.8027 | 0.4744 | 0.4781 | 0.4708 | 0.7059 | 0.4398 | RR=218, RF=245, FR=238, FF=1747 | 2 |
| VID-SPA-02 | ConvNeXt | `convnext_base` | spatial video | video_combined | celeb-df-v2, faceforensics++ | video_only | single | 1 | 8 | 9 | 0.0001 | none | 14096/866/2448 | 0.8566 | 0.7023 | 0.5782 | 0.8942 | 0.9498 | 0.8682 | RR=414, RF=49, FR=302, FF=1683 | 6 |
| VID-SPA-02 | ConvNeXt | `convnext_base` | spatial video | video_combined | celeb-df-v2, faceforensics++ | video_only | single | 1 | 8 | 6 | 0.0001 | weighted_ce | 8648/866/2448 | 0.8717 | 0.6709 | 0.6517 | 0.6911 | 0.8941 | 0.7443 | RR=320, RF=143, FR=171, FF=1814 | 3 |
| VID-SPA-06 | Swin | `swin_base_patch4_window7_224` | spatial video | video_combined | celeb-df-v2, faceforensics++ | video_only | single | 1 | 4 | 8 | 0.0001 | none | 14096/866/2448 | 0.7504 | 0.5253 | 0.4102 | 0.7300 | 0.8145 | 0.5507 | RR=338, RF=125, FR=486, FF=1499 | 5 |
| VID-TMP-01 | ConvNeXt Sequence | `convnext_base` | temporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 8 | 2 | 7 | 0.0001 | none | 14096/866/2448 | 0.8652 | 0.7090 | 0.5991 | 0.8683 | 0.9414 | 0.8445 | RR=402, RF=61, FR=269, FF=1716 | 4 |
| VID-TMP-01 | ConvNeXt Sequence | `convnext_base` | temporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 8 | 2 | 10 | 0.0001 | weighted_ce | 8648/866/2448 | 0.8828 | 0.7092 | 0.6679 | 0.7559 | 0.9217 | 0.8047 | RR=350, RF=113, FR=174, FF=1811 | 7 |
| VID-TMP-02 | ConvNeXt Sequence | `convnext_large` | temporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14096/866/2448 | 0.9089 | 0.7841 | 0.7105 | 0.8747 | 0.9594 | 0.8924 | RR=405, RF=58, FR=165, FF=1820 | 10 |
| VID-TMP-07 | ConvNeXt LSTM | `convnext_large` | temporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 7 | 5e-05 | none | 14140/872/2462 | 0.8794 | 0.7336 | NA | NA | NA | NA | NA | 4 |
| VID-TMP-07 | ConvNeXt LSTM | `convnext_large` | temporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 4 | 2 | 9 | 5e-05 | none | 14096/866/2448 | 0.8779 | 0.7274 | NA | NA | NA | NA | NA | 6 |
| VID-TMP-08 | ConvNeXt Temporal Transformer | `convnext_large` | temporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 7 | 5e-05 | none | 14140/872/2462 | 0.8733 | 0.7249 | NA | NA | NA | NA | NA | 4 |
| VID-TMP-09 | ConvNeXt TCN | `convnext_large` | temporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14140/872/2462 | 0.9102 | 0.7831 | NA | NA | NA | NA | NA | 10 |
| VID-ST-02 | ConvNeXt Hybrid | `convnext_base` | spatiotemporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 12 | 1 | 8 | 0.0001 | none | 14096/866/2448 | 0.7663 | 0.5744 | 0.4381 | 0.8337 | 0.8586 | 0.6036 | RR=386, RF=77, FR=495, FF=1490 | 5 |
| VID-ST-03 | ConvNeXt Hybrid | `convnext_large` | spatiotemporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14096/866/2448 | 0.9089 | 0.7841 | 0.7105 | 0.8747 | 0.9594 | 0.8924 | RR=405, RF=58, FR=165, FF=1820 | 10 |
| VID-ST-05 | MaxViT Hybrid | `maxvit_base_tf_224.in1k` | spatiotemporal video | video_combined | celeb-df-v2, faceforensics++ | video_only | sequence | 4 | 2 | 5 | 5e-05 | none | 14096/866/2448 | 0.8186 | 0.6414 | 0.5123 | 0.8575 | 0.9161 | 0.7482 | RR=397, RF=66, FR=378, FF=1607 | 2 |
| VID-ST-07 | ConvNeXt ConvLSTM | `convnext_large` | spatiotemporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14140/872/2462 | 0.8786 | 0.7342 | NA | NA | NA | NA | NA | 10 |
| VID-ST-08 | ConvNeXt Hybrid Transformer | `convnext_large` | spatiotemporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14140/872/2462 | 0.8956 | 0.7591 | NA | NA | NA | NA | NA | 9 |
| VID-ST-09 | ConvNeXt Hybrid TCN | `convnext_large` | spatiotemporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | video_only | sequence | 4 | 2 | 10 | 5e-05 | none | 14140/872/2462 | 0.8964 | 0.7632 | NA | NA | NA | NA | NA | 7 |

## Dataset Split Table

| Category | Scope | Datasets | Train | Validation | Test | Notes |
|---|---|---|---:|---:|---:|---|
| image | image_combined | cifake, ai-generated-images-vs-real-images | 126094 | 22199 | 31997 | shared by all completed image runs |
| spatial video | video_combined | celeb-df-v2, faceforensics++ | 14096 | 866 | 2448 | default completed spatial split |
| temporal video | video_combined | celeb-df-v2, faceforensics++ | 14096 | 866 | 2448 | default completed temporal split |
| spatiotemporal video | video_combined | celeb-df-v2, faceforensics++ | 14096 | 866 | 2448 | default completed spatiotemporal split |
| temporal/spatiotemporal video | video_all | celeb-df-v2, faceforensics++, real-ai-videos | 14140 | 872 | 2462 | completed new temporal-head split |
| video ablation subset | video_combined | celeb-df-v2, faceforensics++ | 8648 | 866 | 2448 | older weighted/focal ablation split visible in artifacts |

## Training Configuration Table

All completed non-smoke runs use `AdamW` and `CosineAnnealingLR` with warmup, based on `train/image/image_train.py` and `train/video/video_train_backbone.py`.

| ID | Optimizer | Scheduler | LR | Weight decay | Batch | Seq | Epochs configured/run | Early stopping | Imbalance strategy | Seed |
|---|---|---|---:|---:|---:|---:|---|---|---|---:|
| IMG-EXP-01 | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 64 | NA | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| IMG-EXP-02 | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 16 | NA | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| IMG-EXP-04 | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 32 | NA | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| IMG-EXP-05 | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 16 | NA | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| IMG-EXP-07 | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 32 | NA | 10/8 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-SPA-02 focal | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 4 | 1 | 10/5 | patience=3, min_delta=0.0001 | focal_weighting + ablation split | 42 |
| VID-SPA-02 none | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 8 | 1 | 10/9 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-SPA-02 weighted_ce | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 8 | 1 | 10/6 | patience=3, min_delta=0.0001 | loss_weighting + ablation split | 42 |
| VID-SPA-06 | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 4 | 1 | 10/8 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-TMP-01 none | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 2 | 8 | 10/7 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-TMP-01 weighted_ce | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 2 | 8 | 10/10 | patience=3, min_delta=0.0001 | loss_weighting + ablation split | 42 |
| VID-TMP-02 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-TMP-07 video_all | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/7 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-TMP-07 video_combined | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/9 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-TMP-08 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/7 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-TMP-09 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-ST-02 | AdamW | CosineAnnealingLR + warmup | 0.0001 | 0.0001 | 1 | 12 | 10/8 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-ST-03 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-ST-05 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/5 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-ST-07 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-ST-08 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |
| VID-ST-09 | AdamW | CosineAnnealingLR + warmup | 5e-05 | 0.0001 | 2 | 4 | 10/10 | patience=3, min_delta=0.0001 | train_oversample | 42 |

## Model Complexity Table

Parameter counts are estimates from the architecture names, config `params_note`, and the visible model code. Backbones are not frozen in completed configs, so trainable parameters are treated as approximately equal to total parameters.

FLOPs/MACs are marked unavailable because the repository does not contain a completed profiler artifact, and safe FLOP computation would require instantiating each timm model and tracing exact image/clip tensor shapes with an external complexity tool. That would be a new analysis run rather than a completed artifact.

| Model / run family | Approx total params | Approx trainable params | FLOPs/MACs per forward | Basis |
|---|---:|---:|---|---|
| ViT-Base `vit_base_patch16_224` | ~86M | ~86M | not available | standard timm ViT-Base scale; config `params_note` |
| ViT-Large `vit_large_patch16_224` | ~304M | ~304M | not available | standard timm ViT-Large scale; config `params_note` |
| ConvNeXt-Base `convnext_base` | ~89M | ~89M | not available | config `params_note`; timm ConvNeXt-Base scale |
| ConvNeXt-Large `convnext_large` | ~198M | ~198M | not available | config `params_note`; timm ConvNeXt-Large scale |
| Swin-Base `swin_base_patch4_window7_224` | ~88M | ~88M | not available | standard timm Swin-Base scale; config `params_note` |
| MaxViT-Base `maxvit_base_tf_224.in1k` | ~119M | ~119M | not available | standard timm MaxViT-Base scale; config `params_note` |
| ConvNeXt-Large + mean/linear sequence head | ~198M + ~0.003M | ~198M + ~0.003M | not available | `TimmVideoClassifier`: `nn.Linear(feature_dim, 2)` |
| ConvNeXt-Large + LSTM head | ~198M + ~4.2M | ~202M | not available | `nn.LSTM(input=1536, hidden=512, layers=1)` plus linear head |
| ConvNeXt-Large + TCN head | ~198M + ~3.1M | ~201M | not available | two Conv1d layers plus linear head in `TemporalConvHead` |
| ConvNeXt-Large + Transformer head | ~198M + ~11M | ~209M | not available | one TransformerEncoderLayer with `d_model=1536`, default heads/FFN from config/code |
| ConvNeXt-Large + ConvLSTM head | ~198M + ~9.4M | ~207M | not available | ConvLSTM gate conv with `input_dim=1536`, `hidden_dim=512`, kernel 3 |

## Repeated-Seed Analysis

No repeated random-seed experiment set was found in completed artifacts.

Evidence:

- all completed configs with a `seed` field use `seed=42`
- repeated experiment IDs in completed artifacts correspond to changed loss mode, dataset scope, or architecture head, not replicated seed sweeps
- no artifact naming pattern such as `seed_1`, `seed_2`, `run_1`, or alternate seed-specific summaries was found under completed run folders

Therefore, mean +/- standard deviation for accuracy, F1, and ROC-AUC cannot be computed from completed artifacts.

Revision-ready limitation statement:

> All reported values are from deterministic single-run checkpoint evaluation using the saved best checkpoint for each experiment. Repeated-seed analysis was not found in the completed artifacts and should be listed as future work / limitation.

## Cross-Dataset Evaluation Check

No full train-on-one-dataset/test-on-another evaluation was found in completed artifacts.

Evidence:

- completed image runs use `dataset_scope=image_combined` with the same listed datasets for training/validation/test.
- completed `video_combined` runs use Celeb-DF v2 plus FaceForensics++ inside the same split protocol.
- completed `video_all` runs use Celeb-DF v2, FaceForensics++, and Real-AI-Videos inside the same split protocol.
- no completed artifact records a separate train dataset scope and test dataset scope.

Revision-ready limitation statement:

> No full train-on-one-dataset/test-on-another evaluation was found in completed artifacts.

## Best Checkpoint and Evaluation Basis

All completed runs report `best_epoch` and `best_checkpoint` in `final_summary.json`.

Checkpoint selection:

- best metric: `val_f1`
- early stopping: `patience=3`, `min_delta=0.0001`
- saved checkpoint used for final test evaluation: `best.pth`

The `last.pth` checkpoint exists for many runs but should not be used as the reported result unless explicitly re-evaluated. The reported values in this audit are tied to the completed `final_summary.json` and, where available, `test_evaluation.json`.

## Revision-Support Conclusions

- The strongest completed image run is `IMG-EXP-04` ConvNeXt-Base with test F1 `0.9863` and test accuracy `0.9863`.
- The strongest completed video F1 remains `VID-TMP-02` / `VID-ST-03` on `video_combined`, both with test F1 `0.7841`.
- The strongest completed video accuracy is `VID-TMP-09` ConvNeXt TCN on `video_all`, with test accuracy `0.9102` and test F1 `0.7831`.
- The newest temporal-head comparison supports TCN as the strongest added temporal head among LSTM, temporal Transformer, TCN, ConvLSTM, hybrid Transformer, and hybrid TCN.
- Detailed ROC-AUC, average precision, precision, recall, and confusion matrices are available for older completed runs with `test_evaluation.json`; they are not available for newer temporal-head runs unless those evaluations are generated later.
- Repeated-seed and cross-dataset generalization analysis are not present in completed artifacts and should be reported as limitations or future work.
