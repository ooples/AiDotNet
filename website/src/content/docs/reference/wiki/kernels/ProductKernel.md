---
title: "ProductKernel<T>"
description: "Implements a Product kernel that combines multiple kernels by multiplying their outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements a Product kernel that combines multiple kernels by multiplying their outputs.

## For Beginners

The Product kernel combines multiple kernels by multiplying their
similarity scores. This creates interactions between different pattern types.

In mathematical terms: k_prod(x, x') = k1(x, x') × k2(x, x') × ... × kn(x, x')

## How It Works

Why use Product kernels?

1. **Scaling**: Multiply a kernel by a Constant kernel to scale its output
- ConstantKernel(c) × RBF = Scaled RBF kernel

2. **Interaction effects**: When patterns only appear under certain conditions
- RBF × Periodic: Smooth patterns that also vary periodically

3. **Automatic Relevance Determination (ARD)**: Create dimension-specific scaling

by multiplying kernels over different feature subsets

4. **Non-stationary patterns**: Combine with location-dependent kernels

Key difference from Sum kernel:

- Sum: Each kernel contributes independently (additive decomposition)
- Product: Kernels interact multiplicatively (AND-like combination)

Example: RBF × Periodic means "similar if BOTH spatially close AND at similar phase"
(requires both conditions), while RBF + Periodic means "similar if EITHER close OR
at similar phase" (either condition helps).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProductKernel(IKernelFunction<>[])` | Initializes a new Product kernel from an array of kernels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Product kernel value between two vectors. |
| `GetKernels` | Gets the component kernels in this product. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_kernels` | The list of kernels to multiply together. |
| `_numOps` | Operations for performing numeric calculations with type T. |

