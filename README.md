# Diabetes Risk

A PyTorch pipeline for binary diabetes classification from tabular diagnostic data. It compares SGD and Adam, uses SMOTE to address class imbalance, and evaluates the models with ROC-AUC, confusion matrices, and training curves.

[![Notebook checks](https://github.com/SameetAsadullah/diabetes-risk/actions/workflows/quality.yml/badge.svg)](https://github.com/SameetAsadullah/diabetes-risk/actions/workflows/quality.yml)

The notebook is a research analysis of the included datasets; it does not provide a validated clinical diagnostic service.

---

## 📌 Problem Statement

The analysis examines binary diabetes classification using diagnostic features and compares training choices on imbalanced data. ROC-AUC and confusion matrices make the model comparison visible alongside accuracy.

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

## Evaluation flow

The notebook splits the diagnostic records into training and test sets, applies SMOTE to the training set, and compares SGD and Adam during training. It reports a confusion matrix for test predictions and an ROC curve with AUC from predicted probabilities.

## Repository layout

- `notebooks/diabetes-risk.ipynb` — preprocessing, training, and evaluation pipeline
- `data/` — clinical datasets used by the notebook
- `docs/reference-paper.pdf` — supporting reference material

## Run the analysis

Start Jupyter from the notebook directory so its relative data paths resolve consistently:

```bash
cd notebooks
jupyter lab diabetes-risk.ipynb
```
