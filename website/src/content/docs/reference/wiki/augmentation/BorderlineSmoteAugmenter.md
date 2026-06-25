---
title: "BorderlineSmoteAugmenter<T>"
description: "Implements Borderline-SMOTE for imbalanced datasets, focusing on samples near the decision boundary."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Implements Borderline-SMOTE for imbalanced datasets, focusing on samples near the decision boundary.

## For Beginners

Borderline-SMOTE is an improvement over standard SMOTE that only
generates synthetic samples from minority instances that are near the decision boundary
(i.e., in "danger" zones where they have majority class neighbors). This focuses synthetic
data generation where it matters most.

## How It Works

**How it works:**

- For each minority sample, find its k nearest neighbors from BOTH classes
- Classify each minority sample as:

- SAFE: Most neighbors are minority class (not used for synthesis)
- DANGER: Half to most neighbors are majority class (used for synthesis)
- NOISE: All neighbors are majority class (ignored)
- Only generate synthetic samples from DANGER samples

**Borderline-SMOTE Variants:**

- Borderline-SMOTE1: Synthetic samples interpolate only between DANGER samples and minority neighbors
- Borderline-SMOTE2: Synthetic samples can also interpolate toward majority neighbors

This implementation uses Borderline-SMOTE1 by default, configurable via `UseBorderline2`.

**When to use:**

- When standard SMOTE generates too many samples in easy regions
- When you want to focus on the challenging boundary between classes
- When minority samples near the boundary are most important

**Reference:** Han et al., "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning" (2005)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BorderlineSmoteAugmenter(Int32,Int32,Double,Boolean,Double)` | Creates a new Borderline-SMOTE augmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KNeighbors` | Gets the number of nearest neighbors to consider. |
| `MNeighbors` | Gets the number of minority neighbors to use for interpolation. |
| `SamplingRatio` | Gets the sampling ratio for synthetic sample generation. |
| `UseBorderline2` | Gets whether to use Borderline-SMOTE2 (can interpolate toward majority). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `ClassifyMinoritySamples(Matrix<>,Matrix<>)` | Classifies each minority sample as SAFE, DANGER, or NOISE. |
| `ComputeDistance(Matrix<>,Int32,Matrix<>,Int32,Int32)` | Computes the Euclidean distance between two samples. |
| `ComputeDistanceMatrix(Matrix<>)` | Computes the distance matrix for all pairs of samples. |
| `GenerateSyntheticSamples(Matrix<>,Matrix<>,AugmentationContext<>)` | Applies Borderline-SMOTE to generate synthetic samples for the minority class. |
| `GetKNearestNeighbors(Double[0:,0:],Int32,Int32)` | Gets the k nearest neighbors for a given sample. |
| `GetNearestMajorityNeighbor(Matrix<>,Int32,Matrix<>,AugmentationContext<>)` | Gets the nearest majority neighbor for Borderline-SMOTE2. |
| `GetParameters` |  |

