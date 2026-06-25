---
title: "SumKernel<T>"
description: "Implements a Sum kernel that combines multiple kernels by adding their outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements a Sum kernel that combines multiple kernels by adding their outputs.

## For Beginners

The Sum kernel allows you to combine multiple kernels by adding
their similarity scores together. This creates a more expressive model that can capture
different types of patterns simultaneously.

In mathematical terms: k_sum(x, x') = k1(x, x') + k2(x, x') + ... + kn(x, x')

## How It Works

Why use Sum kernels?

1. **Multiple pattern types**: Combine a smooth kernel (RBF) with a periodic kernel

to model data with both smooth trends and seasonal patterns.

2. **Additive structure**: If your function can be decomposed into additive components,

each handled by a different kernel.

3. **Noise modeling**: Add a WhiteNoise kernel to model observation noise.

Example combinations:

- RBF + WhiteNoise: Smooth function with measurement noise
- RBF + Periodic: Smooth trend with seasonal variation
- Linear + RBF: Linear trend with non-linear deviations
- RBF(short) + RBF(long): Capture both local and global patterns

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SumKernel(IKernelFunction<>[])` | Initializes a new Sum kernel from an array of kernels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Sum kernel value between two vectors. |
| `GetKernels` | Gets the component kernels in this sum. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_kernels` | The list of kernels to sum together. |
| `_numOps` | Operations for performing numeric calculations with type T. |

