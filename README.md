# CSI-based-Human-Presence-Detection

This project detects human presence by using WiFi CSI (Channel State Information) magnitude data. It applies signal processing and One-Class SVM (OC-SVM) to classify occupancy based on anomalies learned from empty-room data.

---

## Overview

- **Preprocessing**: Transforms raw CSI into clean, log-scaled frequency-domain data  
- **Training**: Trains OC-SVM on empty-room CSI  
- **Testing**: Detects human presence by identifying outliers in new CSI  
- **Evaluation**: Computes Accuracy, Precision, Recall, and F1-Score (before & after post-processing)  
- **Reference**: Based on the paper "One-Class Support Vector Machine for WiFi-based Device-free Indoor Presence Detection" ([IEEE Xplore](https://ieeexplore.ieee.org/document/10461405))  
- **Dataset**:  
  - Raw CSI: https://utdallas.box.com/s/ln5yo3p7bgz8nbytig0j1tr8vpo6oh8l  
  - Preprocessed CSI: https://utdallas.box.com/s/8uchm97vdem51muk8ykt5bci8gbxiuf0

---

## How to Run

### Preprocess CSI (Python)

```bash
python preprocessing.py
```
Run train.m and test.m to get the results. It will saved in the results folder.
