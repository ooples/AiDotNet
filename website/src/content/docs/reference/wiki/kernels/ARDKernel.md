---
title: "ARDKernel<T>"
description: "Implements the Automatic Relevance Determination (ARD) kernel with per-dimension length scales."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Automatic Relevance Determination (ARD) kernel with per-dimension length scales.

## For Beginners

The ARD (Automatic Relevance Determination) kernel is a powerful extension
of the standard RBF kernel that uses a different length scale for each input dimension.

In mathematical terms: k(x, x') = σ² × exp(-0.5 × Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)

Where:

- σ² is the overall variance (signal variance)
- lᵢ is the length scale for dimension i

## How It Works

Why is ARD important?

1. **Feature Selection**: The length scales tell you which features are important:
- Small lᵢ → Feature i is very relevant (small changes matter a lot)
- Large lᵢ → Feature i is less relevant (can be ignored)
- Very large lᵢ → Feature i is essentially irrelevant

2. **Dimensionality Handling**: In high-dimensional problems, not all features matter equally.

ARD automatically figures out which ones are important.

3. **Interpretability**: After training, inspect the length scales to understand

which features drive your predictions.

4. **Regularization**: By learning length scales, the model avoids overfitting

to irrelevant features.

How it works:

- Start with some initial length scales
- Optimize them by maximizing the log marginal likelihood
- Features with large optimized length scales can be pruned

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ARDKernel(Double[],Double)` | Initializes a new ARD kernel with specified length scales. |
| `ARDKernel(Int32,Double,Double)` | Initializes a new ARD kernel with uniform length scales. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the ARD kernel value between two vectors. |
| `GetLengthScales` | Gets the current length scales. |
| `GetVariance` | Gets the signal variance. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lengthScales` | The length scales for each input dimension. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_variance` | The signal variance (overall scale of the kernel). |

