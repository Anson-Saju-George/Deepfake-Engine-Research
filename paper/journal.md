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
- Datasets used  
- Evaluation metrics:
  - Accuracy  
  - F1 Score  
  - ROC-AUC  

---

### 7.2 Image Model Results
- Training curves  
- Model comparison  

---

### 7.3 Video Model Results
- Spatial vs temporal comparison  
- Performance improvements  

---

### 7.4 Evaluation Analysis
- ROC curves  
- Confusion matrices  

---

### 7.5 Generalization Analysis
- Cross-dataset evaluation  
- Generalization gap  

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