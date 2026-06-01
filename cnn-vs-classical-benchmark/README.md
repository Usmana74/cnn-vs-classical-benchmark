# Classical vs. Deep Learning for Image Recognition: An Empirical Study of Distribution Shift

> **How does the accuracy gap between classical feature-engineering pipelines and deep convolutional networks scale with dataset complexity?**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Report-IEEE%20Format-red.svg)](IEEE_format_report.pdf)

## Overview

This repository contains the full implementation of an empirical benchmarking study comparing **classical feature-engineering pipelines** (HOG, SIFT+BoVW) against a **custom deep CNN** across two image classification benchmarks of increasing complexity: **Fashion-MNIST** and **CIFAR-10**.

The central finding: the accuracy gap between classical and deep pipelines widens **non-linearly** as scene complexity increases — from **1.5 percentage points** on structured Fashion-MNIST to **12.7 percentage points** on naturalistic CIFAR-10 — confirming that hierarchical feature learning confers a disproportionate generalisation advantage in naturalistic scenes.

Written up in **IEEE conference format** (see [`IEEE_format_report.pdf`](IEEE_format_report.pdf)).

---

## Key Results

| Model | Feature Pipeline | Fashion-MNIST Acc. | CIFAR-10 Acc. |
|-------|-----------------|-------------------|---------------|
| **CNN** (SmallCNN) | Raw Pixels | **89.45%** | **66.40%** |
| SVM | HOG + PCA | **89.10%** | 53.70% |
| SVM | SIFT + BoVW + PCA | 83.20% | 49.10% |
| MLP | HOG + PCA | 87.60% | 51.30% |
| Logistic Regression | HOG + PCA | 85.40% | 48.90% |
| KNN | HOG + PCA | 82.10% | 42.30% |
| Naïve Bayes | HOG + PCA | 74.30% | 38.60% |

**Gap on Fashion-MNIST (CNN vs best classical): 0.35 pp → negligible**
**Gap on CIFAR-10 (CNN vs best classical): 12.7 pp → non-linear scaling confirmed**

---

## Architecture

### SmallCNN (Custom PyTorch)
```
Input (32×32×3/1)
  → Conv2d(32) + BN + ReLU + MaxPool
  → Conv2d(64) + BN + ReLU + MaxPool
  → Conv2d(128) + BN + ReLU + MaxPool
  → Dropout(p=0.25) [spatial]
  → FC(256) + ReLU
  → FC(num_classes)
```
Training: Adam (lr=1e-3), label smoothing ε=0.1, 20 epochs, batch size 128.

### Classical Pipelines
- **HOG + PCA(100)**: 9 orientations, 4×4 px/cell, 2×2 cells/block, L2-Hys normalisation
- **SIFT + BoVW + PCA(100)**: vocabulary size 64, MiniBatchKMeans, L1-normalised histograms
- All classical models: 5-fold stratified cross-validation, full hyperparameter grid search

---

## Methodology

- **Datasets**: Fashion-MNIST (10 classes, 60k/10k train/test) and CIFAR-10 (10 classes, 50k/10k)
- **Evaluation**: Accuracy, weighted F1, confusion matrix, ROC curves (OvR), 5-fold stratified CV
- **Reproducibility**: Fixed seed (42), low CV variance (±0.005) confirms stable generalisation
- **Hardware**: Google Colab (T4 GPU)

---

## Repository Structure

```
cnn-vs-classical-benchmark/
├── benchmark_study.ipynb     # Full Colab notebook (recommended entry point)
├── benchmark_study.py        # Plain Python export of the same pipeline
├── IEEE_format_report.pdf    # Full write-up in IEEE conference format
├── latex_report.tex          # LaTeX source for the report
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Usmana74/cnn-vs-classical-benchmark
cd cnn-vs-classical-benchmark
pip install -r requirements.txt
jupyter notebook benchmark_study.ipynb
```

Or open directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Usmana74/cnn-vs-classical-benchmark/blob/main/benchmark_study.ipynb)

---

## Findings & Discussion

The key empirical finding of this study is a **non-linear widening** of the classical-vs-deep accuracy gap as dataset complexity increases:

- On **Fashion-MNIST** (structured, low-texture, centred objects): SVM+HOG+PCA achieves 89.10% vs CNN's 89.45% — a negligible gap of 0.35 pp. Classical feature engineering is competitive when images are clean and structured.
- On **CIFAR-10** (naturalistic scenes, texture-rich, diverse backgrounds): SVM+HOG+PCA drops to 53.70% while CNN achieves 66.40% — a gap of 12.7 pp. The drop is non-linear and confirms that **hierarchical feature learning** is the critical factor for robustness in naturalistic distributions.

This finding motivates further investigation into distribution shift, domain generalisation, and the role of learned representations in out-of-distribution robustness.

---

## Citation

If you use this code or findings in your work, please cite:

```bibtex
@misc{ahmad2026cnnclassical,
  author = {Ahmad, Mohammad Usman},
  title  = {Classical vs. Deep Learning for Image Recognition: An Empirical Study of Distribution Shift},
  year   = {2026},
  url    = {https://github.com/Usmana74/cnn-vs-classical-benchmark}
}
```

---

## Author

**Mohammad Usman Ahmad**
BS Computer Science (AI/CV), PMAS Arid Agriculture University, Pakistan
CGPA: 3.8/4.0

[GitHub](https://github.com/Usmana74) · [LinkedIn](https://linkedin.com/in/usman-ahmad-297b63262) · [PyPI — dataaudit](https://pypi.org/project/dataaudit/)
