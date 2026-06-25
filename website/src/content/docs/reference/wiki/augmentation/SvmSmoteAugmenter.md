---
title: "SvmSmoteAugmenter<T>"
description: "Implements SVM-SMOTE for imbalanced datasets, using SVM decision boundary to identify borderline samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Implements SVM-SMOTE for imbalanced datasets, using SVM decision boundary to identify borderline samples.

## For Beginners

SVM-SMOTE is an enhancement over standard SMOTE that uses Support Vector Machine
classification to identify the most informative samples near the decision boundary. It generates synthetic
samples from support vectors (the most difficult cases), focusing oversampling effort where it matters most.

## How It Works

**How it works:**

- Train an SVM classifier on the data to identify support vectors
- Identify minority samples that are support vectors (near the decision boundary)
- Generate synthetic samples by interpolating between support vector minority samples and their neighbors

**When to use:**

- When the decision boundary is critical for classification
- When you want synthetic samples focused on difficult cases
- When standard SMOTE generates too many samples in easy regions

**Reference:** Nguyen et al., "Borderline Over-Sampling for Imbalanced Data Classification" (2011)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SvmSmoteAugmenter(Int32,Double,Double,Int32,Double)` | Creates a new SVM-SMOTE augmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KNeighbors` | Gets the number of nearest neighbors to use for interpolation. |
| `MaxIterations` | Gets the maximum iterations for SVM training. |
| `SamplingRatio` | Gets the sampling ratio for synthetic sample generation. |
| `SvmC` | Gets the SVM regularization parameter (C). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `ComputeDistanceMatrix(Matrix<>)` | Computes the distance matrix for all pairs of samples. |
| `GenerateSyntheticSamples(Matrix<>,Matrix<>,AugmentationContext<>)` | Applies SVM-SMOTE to generate synthetic samples for the minority class. |
| `GetKNearestNeighbors(Double[0:,0:],Int32,Int32)` | Gets the k nearest neighbors for a given sample. |
| `GetParameters` |  |
| `IdentifySupportVectors(Matrix<>,Matrix<>,AugmentationContext<>)` | Identifies support vectors using a simplified linear SVM approach. |

