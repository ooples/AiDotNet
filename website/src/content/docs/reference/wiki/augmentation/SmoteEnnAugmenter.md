---
title: "SmoteEnnAugmenter<T>"
description: "Implements SMOTE-ENN combination for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Implements SMOTE-ENN combination for imbalanced datasets.

## For Beginners

SMOTE-ENN combines SMOTE oversampling with Edited Nearest Neighbors (ENN)
cleaning. ENN removes samples whose class differs from the majority of their k nearest neighbors,
cleaning both majority and minority samples to improve class separation.

## How It Works

**How it works:**

- Apply SMOTE to generate synthetic minority samples
- For each sample, find its k nearest neighbors
- If the majority of neighbors have a different class, remove the sample

**Benefits over SMOTE-Tomek:**

- More aggressive cleaning than Tomek links
- Removes both misclassified majority AND minority samples
- Better noise removal in overlapping regions

**Reference:** Batista et al., "A Study of the Behavior of Several Methods for Balancing
Machine Learning Training Data" (2004)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SmoteEnnAugmenter(Int32,Int32,Double,Double)` | Creates a new SMOTE-ENN augmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnnKNeighbors` | Gets the number of neighbors used by ENN. |
| `SamplingRatio` | Gets the SMOTE sampling ratio. |
| `SmoteKNeighbors` | Gets the number of nearest neighbors used by SMOTE. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `ApplyEnnCleaning(Matrix<>,Vector<>)` | Applies Edited Nearest Neighbors cleaning to remove noisy samples. |
| `ApplySmoteEnn(Matrix<>,Vector<>,AugmentationContext<>)` | Applies SMOTE-ENN to a labeled dataset. |
| `GetParameters` |  |

