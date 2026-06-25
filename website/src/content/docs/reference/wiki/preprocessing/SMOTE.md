---
title: "SMOTE<T>"
description: "Implements SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets.

## For Beginners

SMOTE works by creating "fake" minority class samples that are similar to
real ones. Here's how it works:

1. Pick a minority class sample
2. Find its k nearest neighbors (other minority samples that are similar)
3. Randomly select one of these neighbors
4. Create a new sample somewhere on the line between the original and the neighbor

Imagine you have two fraud examples at positions [1, 2] and [3, 4].
SMOTE might create a new sample at [2, 3] - right in the middle!
Or at [1.5, 2.5] - one quarter of the way between them.

This is better than just duplicating existing samples because:

- It creates diverse samples that help the model generalize
- The synthetic samples are realistic (they're combinations of real ones)
- It fills in the feature space around minority samples

When to use SMOTE:

- Binary or multi-class classification with imbalanced data
- When you have enough minority samples to find meaningful neighbors (at least k+1)
- When features are numeric (SMOTE doesn't work well with categorical features)

References:

- Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"

## How It Works

SMOTE creates synthetic minority class samples by interpolating between existing minority samples
and their nearest neighbors. It was introduced by Chawla et al. in 2002 and is one of the most
widely used techniques for handling imbalanced data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SMOTE(Double,Int32,Nullable<Int32>)` | Initializes a new instance of the SMOTE class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this oversampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateSyntheticSamples(Matrix<>,List<Int32>,Int32)` | Generates synthetic samples using SMOTE interpolation. |
| `InterpolateSamples(Vector<>,Vector<>)` | Creates a synthetic sample by interpolating between two samples. |

