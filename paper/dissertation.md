# 🎓 FINAL DISSERTATION STRUCTURE (COMPLETE)

---

## 1. Abstract
- Problem: Deepfake detection across images and videos  
- Approach: Multi-modal pipeline (image → video → temporal)  
- Key findings (e.g., ConvNeXt best, temporal improves performance)  
- Contributions  

---

## 2. Introduction

### 2.1 Background
- Deepfakes and their impact  

### 2.2 Problem Statement
- Image vs video complexity  
- Dataset and methodological challenges  

### 2.3 Objectives
- Build a research-grade pipeline  
- Compare image vs video models  
- Analyze spatial vs temporal learning  

### 2.4 Contributions
- Clean dataset pipeline  
- Protocol-based experimentation  
- Comparative evaluation  

---

## 3. Literature Review

### 3.1 Image-Based Deepfake Detection  
### 3.2 Video-Based Deepfake Detection  
### 3.3 Temporal Modeling Approaches  
### 3.4 CNN vs Transformer Architectures  

---

## 4. Problem Formulation & System Overview

### 4.1 Task Definition
- Binary classification (fake vs real)

### 4.2 Modalities
- Image (spatial)  
- Video (single-frame)  
- Video (sequence)  

### 4.3 System Pipeline Overview
- Include system architecture diagram  

---

## 5. Dataset and Data Pipeline

### 5.1 Dataset Description
- CIFAKE  
- AI-generated vs real  
- Celeb-DF  
- FaceForensics++  

### 5.2 Data Cleaning
- Removed corrupted samples  
- Validation tools  

### 5.3 Data Preprocessing
- Image transforms  
- Video sampling  
- Frame extraction (optional)  

### 5.4 Exploratory Data Analysis
- Class imbalance  
- Dataset bias  

---

## 6. Feature Representation

### 6.1 Image Representation
- RGB tensors  

### 6.2 Video Representation
- Single frame (spatial baseline)  
- Sequence (temporal learning)  

### 6.3 Representation Design Choices
- Why temporal learning matters  

---

## 7. Methodology (CORE CHAPTER)

### 7.1 Protocol Design
- image_only  
- video_only  
- frame_only  

### 7.2 Model Architectures

#### 7.2.1 Image Models
- ViT  
- ConvNeXt  
- Swin  
- DeiT  
- MaxViT  
- EVA  

#### 7.2.2 Video Models

##### (a) Spatial Models
- Single-frame video classification  

##### (b) Temporal Models
- Sequence-based learning  

##### (c) Spatiotemporal Models
- Hybrid approaches  

---

### 7.3 Training Strategy
- Image training pipeline  
- Video training pipeline  
- Sampling strategies  

---

### 7.4 Hyperparameter Setup
- Fixed baseline  
- Controlled tuning  

---

## 8. Experiments and Results

### 8.1 Experimental Setup
All experiments were executed on a Windows 11 system (`10.0.26200`) equipped with an NVIDIA GeForce RTX 5080 Laptop GPU (`15.92 GB` VRAM, driver `595.97`), `63.42 GB` RAM, and a `20`-core Intel CPU. The primary software environment was the Conda environment `torch`, built around Python `3.12.12`, PyTorch `2.10.0+cu130`, TorchVision `0.25.0+cu130`, TorchAudio `2.10.0+cu130`, CUDA `13.0`, cuDNN `91200`, and timm `1.0.25`. Additional core libraries included NumPy `2.3.4`, pandas `2.3.3`, scikit-learn `1.7.2`, OpenCV `4.13.0`, Pillow `12.0.0`, and Matplotlib `3.10.7`.

The active dataset inventory contained `192,016` cleaned raw samples, comprising `179,988` images and `12,028` videos. Image experiments followed the `image_only` protocol and preserved original source train/test boundaries while deriving validation from source training partitions. Video experiments followed the `video_only` protocol with identity-aware grouped splitting at `70/10/20` for train/validation/test. Spatial video baselines used `mode="single"`, whereas temporal and spatiotemporal experiments used `mode="sequence"` with deterministic center-clip sampling at evaluation time.

Evaluation was performed from saved checkpoints through explicit prediction-export scripts that produced `test_predictions.csv` and `test_evaluation.json` for each completed run. Metrics reported in this chapter are accuracy, precision, recall, F1-score, ROC-AUC, average precision, and confusion matrices.

---

### 8.2 Image Model Results

The image-domain results were consistently strong across the completed runs and established the highest-performing branch of the current project. The best completed image experiment was `IMG-EXP-04` (`ConvNeXt-Base`), which reached `0.9863` accuracy, `0.9863` F1, and `0.9968` ROC-AUC on the held-out image test set. `Swin-Base` and `ConvNeXt-Large` remained close behind, while the ViT runs were clearly weaker than the top convolutional and hierarchical-transformer baselines.

| Experiment | Family | Model | Accuracy | F1 | Precision | Recall | ROC-AUC | AP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IMG-EXP-04 | ConvNeXt | convnext_base | 0.9863 | 0.9863 | 0.9867 | 0.9858 | 0.9968 | 0.9956 |
| IMG-EXP-07 | Swin | swin_base_patch4_window7_224 | 0.9842 | 0.9842 | 0.9860 | 0.9823 | 0.9987 | 0.9986 |
| IMG-EXP-05 | ConvNeXt | convnext_large | 0.9840 | 0.9840 | 0.9831 | 0.9849 | 0.9966 | 0.9966 |
| IMG-EXP-01 | ViT | vit_base_patch16_224 | 0.9703 | 0.9702 | 0.9734 | 0.9670 | 0.9939 | 0.9931 |
| IMG-EXP-02 | ViT | vit_large_patch16_224 | 0.9548 | 0.9546 | 0.9588 | 0.9504 | 0.9899 | 0.9893 |

Useful figure outputs for this subsection include:

- `graphs/overview/image_comparison.png`
- `graphs/overview/cnn_vs_transformer.png`
- `graphs/runs/image/image/.../loss_curve.png`
- `graphs/runs/image/image/.../f1_curve.png`

---

### 8.3 Video Model Results

The video-domain results were lower than the image-domain ceiling, but they still revealed a clear ordering across model families and sequence settings. The strongest completed video runs were `VID-TMP-02` (`ConvNeXt-Large` temporal) and `VID-ST-03` (`ConvNeXt-Large` spatiotemporal), both of which achieved `0.9089` accuracy, `0.7841` F1, and `0.9594` ROC-AUC. Among the spatial baselines, `VID-SPA-02 ConvNeXt-Base` with standard cross-entropy (`loss=none`) was strongest.

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

Useful figure outputs for this subsection include:

- `graphs/overview/spatial_comparison.png`
- `graphs/overview/temporal_comparison.png`
- `graphs/overview/spatiotemporal_comparison.png`
- `graphs/overview/loss_function_ablation.png`

---

### 8.4 Evaluation Metrics

The project now exports prediction-based evaluation artifacts for each completed run. This allows confusion matrices, ROC curves, and derived ranking metrics to be computed from held-out per-sample predictions rather than only from aggregate summaries. The image ROC and confusion outputs show that the top image models operate close to the upper-left corner of ROC space, with minimal confusion between real and fake classes. The video ROC and confusion outputs, in contrast, show materially larger overlap between classes and more visible precision-recall trade-offs.

Core reported metrics in this chapter are:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- average precision
- confusion matrix

Relevant generated figures:

- `graphs/overview/image_confusion_matrix.png`
- `graphs/overview/video_confusion_matrix.png`
- `graphs/overview/image_roc_curve.png`
- `graphs/overview/video_roc_curve.png`

---

### 8.5 Analysis

Three findings are most important. First, image detection is far stronger than the current completed video runs, indicating that modality-specific difficulty remains substantial even under a protocol-separated setup. Second, ConvNeXt is the strongest family in the current completed evidence, dominating both image and video leaderboards. Third, temporal aggregation is beneficial: the top temporal and spatiotemporal ConvNeXt-Large runs clearly outperform the spatial video baselines.

One implementation-aware caveat is also important for interpretation. The current completed `VID-TMP-02` and `VID-ST-03` runs are numerically identical, which matches the implementation note in `train/video/video_config.md` that the active temporal and spatiotemporal branches currently share the same sequence trainer mechanics. This means the present evidence should be interpreted as clip-based sequence aggregation rather than as two fundamentally different native-video paradigms. In addition, focal loss underperformed badly on the completed spatial ConvNeXt comparison, so the default oversampling plus standard cross-entropy path remains the strongest reporting baseline in the current codebase.

Relevant summary figures:

- `graphs/overview/leaderboard.png`
- `graphs/overview/generalization_gap.png`
- `graphs/overview/cross_modality_comparison.png`

---

## 9. System Implementation (APPLICATION)

### 9.1 System Architecture
- Model pipeline  

### 9.2 Inference Pipeline
- Image input  
- Video input  

### 9.3 Interface / Usage
- Brief description  

---

## 10. Discussion
- Key findings  
- Limitations  
- Dataset bias  
- Model weaknesses  

---

## 11. Conclusion
- Summary  
- Contributions  
- Future work  

---

## 12. Acknowledgements

---

## 13. References

---

# 📄 Estimated Page Count

## Core Chapters

| Chapter | Pages |
|--------|------|
| Introduction | 4–6 |
| Literature Review | 8–12 |
| Problem Formulation & System Overview | 5–7 |
| Dataset & Data Pipeline | 8–12 |
| Feature Representation | 5–8 |
| Methodology | 12–18 |
| Experiments & Results | 15–25 |
| Discussion | 5–8 |
| Conclusion | 3–5 |

Subtotal: **~65–100 pages**

---

## Supporting Chapters

| Chapter | Pages |
|--------|------|
| System Implementation | 5–10 |
| Acknowledgements | 1 |
| References | 3–6 |

Total: **~75–115 pages**

---

# 🔥 Key Notes

## Your Thesis Flow
1. Define problem  
2. Build pipeline  
3. Compare models  
4. Present results  
5. Explain insights  

---

## Avoid These Mistakes
- Mixing image and video results  
- Overemphasizing the application  

---

## Your Strength
Focus on:
**Methodology + Experimental Rigor**

Not just:
> "We built a deepfake detector"
