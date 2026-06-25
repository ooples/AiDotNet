---
title: "RationalQuadraticKernel<T>"
description: "Implements the Rational Quadratic kernel, equivalent to an infinite mixture of RBF kernels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Rational Quadratic kernel, equivalent to an infinite mixture of RBF kernels.

## For Beginners

The Rational Quadratic (RQ) kernel is a powerful kernel that can be
thought of as an infinite mixture of RBF kernels with different length scales.

In mathematical terms:
k(x, x') = σ² × (1 + r²/(2αl²))^(-α)

Where:

- r = |x - x'| is the Euclidean distance
- l is the length scale
- α is the "scale mixture" parameter
- σ² is the variance

The α parameter controls how the kernel behaves:

- α → ∞: Approaches the RBF kernel (single length scale)
- Small α: More heavy-tailed (multiple length scales contribute)

## How It Works

Why use Rational Quadratic?

1. **Multi-scale**: Naturally captures patterns at different scales
2. **Robustness**: Less sensitive to length scale choice than RBF
3. **Heavy tails**: Points far apart still have some correlation
4. **Flexibility**: Interpolates between RBF and more flexible behaviors

When to use:

- Data has patterns at multiple scales
- You're not sure what length scale to use
- RBF seems too restrictive

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RationalQuadraticKernel(Double,Double,Double)` | Initializes a new Rational Quadratic kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the scale mixture parameter (α). |
| `LengthScale` | Gets the length scale. |
| `Variance` | Gets the signal variance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Rational Quadratic kernel value between two vectors. |
| `CalculateGradient(Vector<>,Vector<>)` | Computes the gradient of the kernel with respect to the input. |
| `CalculateHyperparameterGradients(Vector<>,Vector<>)` | Computes the gradient with respect to hyperparameters. |
| `EffectiveLengthScales` | Returns the "effective number of length scales" in the mixture. |
| `IsEffectivelyRBF` | Determines if this kernel approximates an RBF kernel. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The scale mixture parameter (α). |
| `_lengthScale` | The length scale parameter. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_variance` | The signal variance. |

