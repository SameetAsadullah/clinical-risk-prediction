# 🔬 Clinical Risk Prediction with PyTorch

This project implements an end-to-end ML pipeline for binary clinical risk classification using PyTorch. The objective is to detect early signs of disease risk from patient diagnostics through scalable and efficient machine learning.

> ⚡ Built with deployability and model evaluation at the forefront — includes optimizer benchmarking, imbalance correction, and visual diagnostics.

---

## 📌 Problem Statement

Medical datasets often suffer from class imbalance and noisy features. The goal is to develop a robust binary classifier that generalizes well on clinical risk detection tasks, especially in skewed datasets, and provides interpretable evaluation metrics.

---

## 🚀 Pipeline Overview

- **Data Preprocessing**: normalization, data cleansing
- **Imbalance Handling**: applied **SMOTE** for minority class oversampling
- **Modeling**: optimized a PyTorch-based classifier using:
  - **SGD** and **Adam** optimizers
  - Comparative learning rate experiments
- **Evaluation Metrics**:
  - Accuracy, Loss Curves
  - ROC-AUC, Confusion Matrix
  - Class distribution visualization

---

## 🛠️ Stack & Tools

- `PyTorch`
- `imbalanced-learn` for SMOTE
- `Scikit-learn`, `Pandas`, `Matplotlib`, `Seaborn`
