---
title: "SmoteAugmenter<T>"
description: "Implements SMOTE (Synthetic Minority Over-sampling Technique) for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Implements SMOTE (Synthetic Minority Over-sampling Technique) for imbalanced datasets.

## For Beginners

SMOTE creates new synthetic samples for the minority class
by interpolating between existing minority samples and their nearest neighbors.
This helps balance imbalanced datasets where one class has far fewer samples than others.

## How It Works

**How it works:**

- For each minority sample, find its k nearest neighbors (also from minority class)
- Randomly select one of these neighbors
- Create a new sample along the line between the original and the neighbor

**When to use:**

- Classification with severe class imbalance (e.g., fraud detection, rare disease)
- When the minority class has too few samples to learn from
- When undersampling the majority class would lose too much information

**When NOT to use:**

- When classes are already balanced
- For regression tasks (use other techniques)
- When features are highly categorical (use SMOTE-NC instead)

**Reference:** Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SmoteAugmenter(Int32,Double,Double)` | Creates a new SMOTE augmenter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KNeighbors` | Gets the number of nearest neighbors to consider. |
| `SamplingRatio` | Gets the sampling ratio for synthetic sample generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `ApplySmoteWithLabels(Matrix<>,Vector<>,AugmentationContext<>)` | Applies SMOTE and returns combined original and synthetic data. |
| `GenerateSyntheticSamples(Matrix<>,AugmentationContext<>)` | Applies SMOTE to generate synthetic samples for the minority class. |
| `GetParameters` |  |

