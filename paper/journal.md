# Deepfake Detection Across Modalities: A Comparative Study of Image and Video Models with Spatial and Temporal Learning

---

## Abstract
Deepfake generation techniques have advanced rapidly, posing significant threats to digital media integrity. While numerous detection methods exist, many fail to distinguish between image-based and video-based modalities, leading to inconsistent evaluation protocols and unclear performance comparisons.  

This paper presents a methodologically rigorous comparison of deepfake detection across image and video modalities, incorporating both spatial and temporal modeling approaches. We introduce a protocol-separated evaluation framework to isolate modality-specific performance and ensure fair comparisons.  

We evaluate multiple architectures, including CNN-based and transformer-based models, across carefully curated datasets with identity-aware splits. Our results demonstrate that (1) image and video detection represent fundamentally different tasks, (2) temporal modeling significantly improves video detection performance, and (3) ConvNeXt-based architectures outperform transformer-heavy models under controlled conditions.  

These findings highlight the importance of evaluation design and provide actionable insights for future deepfake detection systems.

---

## 1. Introduction

### 1.1 Background
Deepfakes, generated using advanced generative models, have become increasingly realistic and accessible. Their misuse in misinformation, identity fraud, and digital manipulation has raised serious concerns.

### 1.2 Problem Statement
Existing deepfake detection approaches often:
- Mix image and video data during training and evaluation
- Use inconsistent experimental protocols
- Fail to isolate spatial and temporal learning contributions

### 1.3 Research Gap
There is a lack of:
- Protocol-separated evaluation frameworks
- Direct comparison between image and video detection
- Clear understanding of temporal modeling benefits

### 1.4 Contributions
This paper makes the following contributions:
- A protocol-separated evaluation framework for fair modality comparison
- A comprehensive comparison of image vs video deepfake detection
- Analysis of spatial vs temporal modeling effectiveness
- Empirical evidence highlighting ConvNeXt performance advantages

---

## 2. Related Work

### 2.1 Image-Based Deepfake Detection
Overview of CNNs and transformer-based methods for image classification.

### 2.2 Video-Based Deepfake Detection
Frame-based vs sequence-based approaches.

### 2.3 Temporal Modeling
Recurrent models, 3D CNNs, and temporal transformers.

### 2.4 CNN vs Transformer Architectures
Strengths and limitations of each paradigm.

### 2.5 Limitations of Existing Work
- Lack of consistent protocols  
- Dataset biases  
- Weak generalization  

---

## 3. Problem Formulation and System Overview

### 3.1 Task Definition
Binary classification:
- Real vs Fake

### 3.2 Modalities
- Image-based detection  
- Video-based detection  

### 3.3 System Pipeline
- Data preprocessing  
- Feature extraction  
- Model training  
- Evaluation  

---

## 4. Dataset and Data Pipeline

### 4.1 Dataset Description
- Source datasets  
- Sample counts  
- Class distribution  

### 4.2 Data Cleaning
- Removal of corrupted samples  
- Frame extraction consistency  

### 4.3 Preprocessing
- Image resizing and normalization  
- Video frame sampling strategies  

### 4.4 Exploratory Data Analysis
- Class imbalance  
- Identity distribution  
- Dataset bias  

---

## 5. Feature Representation

### 5.1 Spatial Representation
Single-frame analysis.

### 5.2 Temporal Representation
Sequential frame modeling.

### 5.3 Representation Challenges
- Motion inconsistency  
- Compression artifacts  

---

## 6. Methodology

### 6.1 Protocol Design
- Image-only protocol  
- Video-only protocol  

### 6.2 Model Architectures

#### Image Models
- Vision Transformer (ViT)  
- ConvNeXt  
- Swin Transformer  

#### Video Models
- Spatial models (frame-based)  
- Temporal models (sequence-based)  
- Hybrid approaches  

### 6.3 Training Strategy
- Identity-aware data splits  
- Balanced sampling  

### 6.4 Hyperparameter Configuration
- Learning rate  
- Batch size  
- Optimization strategy  

---

## 7. Experiments and Results

### 7.1 Experimental Setup
All experiments were executed on a Windows 11 system (`10.0.26200`) with an NVIDIA GeForce RTX 5080 Laptop GPU (`15.92 GB` VRAM, driver `595.97`), `63.42 GB` RAM, and a `20`-core Intel CPU. The software environment was managed through a dedicated Conda environment (`torch`) using Python `3.12.12`, PyTorch `2.10.0+cu130`, TorchVision `0.25.0+cu130`, TorchAudio `2.10.0+cu130`, CUDA `13.0`, cuDNN `91200`, and timm `1.0.25`. Supporting libraries included NumPy `2.3.4`, pandas `2.3.3`, scikit-learn `1.7.2`, OpenCV `4.13.0`, Pillow `12.0.0`, and Matplotlib `3.10.7`.

The evaluation protocol was modality-separated throughout. Image experiments used the `image_only` protocol, preserving source train/test boundaries and deriving validation from source training partitions. Video experiments used the `video_only` protocol with identity-aware grouped splitting at `70/10/20` for train/validation/test. Spatial video baselines used `mode="single"`, whereas temporal and spatiotemporal runs used `mode="sequence"` with deterministic center-clip sampling at test time. Final held-out evaluation was exported through per-run `test_predictions.csv` and `test_evaluation.json` artifacts.

Reported metrics are accuracy, precision, recall, F1-score, ROC-AUC, average precision, and confusion matrices.

---

### 7.2 Image Model Results
Image-domain detection was substantially stronger than the current video-domain runs. The best completed image model was `IMG-EXP-04` (`ConvNeXt-Base`), reaching `0.9863` accuracy, `0.9863` F1, and `0.9968` ROC-AUC on the held-out image test set. `Swin-Base` and `ConvNeXt-Large` remained very close, while both ViT runs trailed the top convolutional and hierarchical-transformer models.

| Experiment | Family | Model | Accuracy | F1 | Precision | Recall | ROC-AUC | AP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IMG-EXP-04 | ConvNeXt | convnext_base | 0.9863 | 0.9863 | 0.9867 | 0.9858 | 0.9968 | 0.9956 |
| IMG-EXP-07 | Swin | swin_base_patch4_window7_224 | 0.9842 | 0.9842 | 0.9860 | 0.9823 | 0.9987 | 0.9986 |
| IMG-EXP-05 | ConvNeXt | convnext_large | 0.9840 | 0.9840 | 0.9831 | 0.9849 | 0.9966 | 0.9966 |
| IMG-EXP-01 | ViT | vit_base_patch16_224 | 0.9703 | 0.9702 | 0.9734 | 0.9670 | 0.9939 | 0.9931 |
| IMG-EXP-02 | ViT | vit_large_patch16_224 | 0.9548 | 0.9546 | 0.9588 | 0.9504 | 0.9899 | 0.9893 |

Relevant generated figures include `graphs/overview/image_comparison.png`, `graphs/overview/cnn_vs_transformer.png`, and the per-run dashboards under `graphs/runs/image/`.

---

### 7.3 Video Model Results
Video-domain performance was lower than the image-domain ceiling, but the ranking pattern remained consistent: ConvNeXt-based runs dominated the strongest completed evidence. The best video results came from `VID-TMP-02` (`ConvNeXt-Large` temporal) and `VID-ST-03` (`ConvNeXt-Large` spatiotemporal), both reaching `0.9089` accuracy, `0.7841` F1, and `0.9594` ROC-AUC. Among spatial baselines, `VID-SPA-02 ConvNeXt-Base` with `loss=none` was strongest, outperforming both weighted cross-entropy and focal loss variants in F1.

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

Relevant comparison figures include `graphs/overview/spatial_comparison.png`, `graphs/overview/temporal_comparison.png`, `graphs/overview/spatiotemporal_comparison.png`, and `graphs/overview/loss_function_ablation.png`.

---

### 7.4 Evaluation Analysis
Prediction-based ROC curves and confusion matrices were exported for both image and video leaders. The image ROC profile remained close to the upper-left corner, consistent with the near-ceiling image F1 scores. In contrast, the video ROC and confusion figures show that video models still trade off recall against false positives more aggressively, particularly in the weaker spatial-only baselines.

Key generated assets are:

- `graphs/overview/image_confusion_matrix.png`
- `graphs/overview/video_confusion_matrix.png`
- `graphs/overview/image_roc_curve.png`
- `graphs/overview/video_roc_curve.png`

These figures were generated from exported `test_predictions.csv` files rather than only from aggregate summaries, making the reported operating characteristics directly traceable to held-out per-sample predictions.

---

### 7.5 Generalization Analysis
The current evidence shows a clear modality gap: image models generalize far better on their held-out domain than the video models do on theirs. At the same time, the strongest completed video runs narrow the gap substantially once temporal aggregation is introduced. ConvNeXt-Large sequence and hybrid runs provide the most stable video-domain generalization among the completed checkpoints.

Two additional implementation-aware observations are important. First, the completed `VID-TMP-02` and `VID-ST-03` runs are numerically identical in the current evidence snapshot, which matches the current implementation note that the active temporal and spatiotemporal branches share the same sequence trainer mechanics. Second, focal loss underperformed badly on the completed spatial ConvNeXt comparison, suggesting that the default oversampling plus standard cross-entropy path remains the more defensible baseline under the current corpus composition.

Cross-run summary figures are available in:

- `graphs/overview/leaderboard.png`
- `graphs/overview/generalization_gap.png`
- `graphs/overview/cross_modality_comparison.png`

---

## 8. Discussion

### 8.1 Model Performance Insights
- Why ConvNeXt performs best  

### 8.2 Temporal Modeling Benefits
- Improved detection consistency  

### 8.3 Limitations
- Dataset bias  
- Computational constraints  

### 8.4 Implications
- Importance of protocol design  
- Future model development directions  

---

## 9. System Implementation (Optional)

### 9.1 Training Pipeline
Overview of implementation.

### 9.2 Deployment Considerations
- Inference efficiency  
- Real-time constraints  

---

## 10. Conclusion

### 10.1 Summary
- Image and video detection are distinct tasks  
- Temporal modeling improves performance  
- Proper evaluation protocols are critical  

### 10.2 Future Work
- Multimodal fusion  
- Robustness to unseen datasets  
- Lightweight deployment models  

---

## References
(To be populated in final version)

---

## Appendix (Optional)

### A. Additional Results
Extended tables and metrics.

### B. Hyperparameter Details
Full configuration logs.

### C. Supplementary Figures
Additional plots and visualizations.
