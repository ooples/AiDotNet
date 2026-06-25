---
title: "DotProductKernel<T>"
description: "Implements the Dot Product (Linear) kernel with optional inhomogeneity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Dot Product (Linear) kernel with optional inhomogeneity.

## For Beginners

The Dot Product kernel measures similarity by computing the inner product
(dot product) between two vectors, optionally with a constant offset.

In mathematical terms: k(x, x') = σ₀² + x · x'

Where:

- σ₀² is the variance of the constant offset (inhomogeneity parameter)
- x · x' is the dot product (sum of element-wise products)

## How It Works

Why use the Dot Product kernel?

1. **Linear relationships**: When your data has a linear relationship, this kernel works well
- Equivalent to Bayesian linear regression when used in a GP

2. **Interpretability**: Each feature's contribution is directly proportional to its value

3. **Computational efficiency**: Very fast to compute, O(d) for d-dimensional vectors

4. **Feature importance**: In a GP with this kernel, the predictive variance grows

indefinitely away from the origin, which can be useful for extrapolation

The inhomogeneity parameter σ₀² adds a constant "bias" to all similarities, which allows
the model to capture a non-zero mean function.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DotProductKernel(Double)` | Initializes a new instance of the Dot Product kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Dot Product kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma0Squared` | The variance of the constant offset (σ₀²). |

