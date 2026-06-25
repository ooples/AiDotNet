---
title: "IUnderSampler<T>"
description: "Interface for undersampling techniques that reduce the majority class."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation.Tabular.Undersampling`

Interface for undersampling techniques that reduce the majority class.

## For Beginners

Undersampling is a technique to handle imbalanced datasets by
removing samples from the majority class. This helps classifiers focus on the minority class
without being overwhelmed by majority samples.

## How It Works

**Common undersampling methods:**

- Random: Randomly remove majority samples
- NearMiss: Remove majority samples far from minority class
- Tomek Links: Remove majority samples forming Tomek links
- Edited Nearest Neighbors: Remove misclassified samples

**Trade-offs:** Undersampling may discard useful information, but reduces
training time and can prevent majority class bias. Often combined with oversampling
(e.g., SMOTE-Tomek, SMOTE-ENN).

## Methods

| Method | Summary |
|:-----|:--------|
| `Undersample(Matrix<>,Vector<>,)` | Performs undersampling on the dataset. |

