---
title: "ConstantKernel<T>"
description: "Implements the Constant kernel, which returns a constant value regardless of input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Constant kernel, which returns a constant value regardless of input.

## For Beginners

The Constant kernel is the simplest possible kernel - it always returns
the same value, no matter what inputs you give it.

In mathematical terms: k(x, x') = c

Where c is a constant value (the "constant value" parameter).

## How It Works

Why would you want a kernel that ignores the input?

The Constant kernel is primarily used as a building block for more complex kernels:

1. **Scaling other kernels**: When you multiply the Constant kernel with another kernel

(like RBF), you can control the overall scale of your predictions.

- ConstantKernel(c) * RBFKernel() gives you a scaled RBF kernel

2. **Adding bias**: When you add a Constant kernel to another kernel, you're essentially

adding a constant offset to all predictions.

- RBFKernel() + ConstantKernel(c) adds a "baseline" to your model

3. **Modeling a constant mean**: In Gaussian Processes, the Constant kernel represents

the assumption that all points have some shared, constant relationship.

By itself, the Constant kernel assumes that all data points are equally similar to each
other - not very useful! But combined with other kernels, it's a powerful tool.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConstantKernel(Double)` | Initializes a new instance of the Constant kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Constant kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_constantValue` | The constant value returned by this kernel for all inputs. |
| `_numOps` | Operations for performing numeric calculations with type T. |

