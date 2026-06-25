---
title: "AdasynAugmenter<T>"
description: "Implements ADASYN (Adaptive Synthetic Sampling) for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Implements ADASYN (Adaptive Synthetic Sampling) for imbalanced datasets.

## For Beginners

ADASYN is an extension of SMOTE that adaptively generates more
synthetic samples in regions where the minority class is harder to learn (i.e., where there
are more majority class neighbors). This focuses the synthetic data where it's most needed.

## How It Works

**How ADASYN differs from SMOTE:**

- SMOTE: Generates the same number of synthetic samples for each minority instance
- ADASYN: Generates more synthetic samples for minority instances that have more majority neighbors

This adaptive approach helps the classifier focus on the hardest-to-learn examples.

**Algorithm:**

- For each minority sample, calculate the ratio of majority neighbors to total neighbors
- Normalize these ratios to get a distribution
- Generate synthetic samples proportionally - more for samples with more majority neighbors

**When to use:**

- When the decision boundary is complex and irregular
- When minority samples near the boundary need more representation
- When standard SMOTE doesn't improve minority class recall enough

**Reference:** He et al., "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning" (2008)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdasynAugmenter(Int32,Double,Double)` | Creates a new ADASYN augmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta` | Gets the target balance ratio between minority and majority classes. |
| `KNeighbors` | Gets the number of nearest neighbors to consider. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `CalculateDifficultyRatios(Matrix<>,Matrix<>,AugmentationContext<>)` | Calculates the difficulty ratio for each minority sample. |
| `ComputeDistance(Matrix<>,Int32,Matrix<>,Int32,Int32)` | Computes the Euclidean distance between two samples. |
| `ComputeDistanceMatrix(Matrix<>)` | Computes the distance matrix for all pairs of minority samples. |
| `GenerateSyntheticSamples(Matrix<>,Matrix<>,AugmentationContext<>)` | Applies ADASYN to generate synthetic samples for the minority class. |
| `GetKNearestNeighbors(Double[0:,0:],Int32,Int32)` | Gets the k nearest neighbors for a given sample. |
| `GetParameters` |  |

