# 🎓 Image Classification with ResNet (PyTorch) and CatBoost

This repository contains two complementary machine learning projects developed for a university coursework — both focusing on **image classification** using different approaches:

- One project uses **deep learning with ResNet architecture in PyTorch**.
- The other project applies **CatBoost (gradient boosting)** on **image features converted to tabular format**.

By comparing deep learning and gradient boosting approaches for the same image classification task, this project highlights the strengths and trade-offs of each method.

---

## 📁 Contents

1. [`Resnet_ModelTrainer.ipynb`](./Resnet_ModelTrainer.ipynb) — **Image classification with PyTorch**

---

## Model 1: Deep Learning with PyTorch & ResNet

### Overview
This notebook provides a full training pipeline for image classification using the ResNet architecture in PyTorch. It includes data preprocessing, augmentation, model training, validation, and evaluation.

### Model Used
- **Architecture**: ResNet50
- **Framework**: PyTorch

### Key Features
- Custom dataset class with bounding box cropping and Albumentations support
- Early stopping & learning rate scheduling
- Evaluation using confusion matrix, classification report, F1 score

### Dependencies
- `torch`, `torchvision`, `albumentations`, `scikit-learn`, `matplotlib`, `numpy`, `PIL`, `pandas`
  
