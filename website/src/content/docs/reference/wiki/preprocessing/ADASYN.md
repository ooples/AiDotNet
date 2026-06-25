---
title: "ADASYN<T>"
description: "Implements ADASYN (Adaptive Synthetic Sampling) for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements ADASYN (Adaptive Synthetic Sampling) for handling imbalanced datasets.

## For Beginners

ADASYN improves on SMOTE by being smarter about WHERE to create
synthetic samples:

1. For each minority sample, look at its k nearest neighbors (from ALL classes)
2. Count how many of those neighbors are majority class samples
3. Minority samples with MORE majority neighbors are "harder to learn"
4. Create MORE synthetic samples near the "hard" minority samples

Example scenario:

- Minority sample A has 4 minority neighbors, 1 majority neighbor → Easy, few synthetics
- Minority sample B has 1 minority neighbor, 4 majority neighbors → Hard, many synthetics

This is better than regular SMOTE because:

- Focuses sampling effort where it's needed most
- Helps the model learn the difficult boundary cases
- Reduces risk of overfitting in easy regions

When to use ADASYN:

- When the boundary between classes is complex
- When some minority samples are "islands" in majority territory
- When you want adaptive, data-driven sampling

References:

- He et al. (2008). "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning"

## How It Works

ADASYN is an extension of SMOTE that adaptively generates synthetic samples based on the
local density of minority samples. It creates more synthetic samples in regions where
the minority class is harder to learn (i.e., surrounded by majority class samples).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ADASYN(Double,Int32,Nullable<Int32>)` | Initializes a new instance of the ADASYN class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this oversampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateSyntheticSamples(Matrix<>,List<Int32>,Int32)` | Generates synthetic samples using ADASYN's adaptive approach. |
| `InterpolateSamples(Vector<>,Vector<>)` | Creates a synthetic sample by interpolating between two samples. |

