---
title: "ExponentialKernel<T>"
description: "Implements the Exponential kernel function for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Exponential kernel function for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Exponential kernel is like a "similarity detector" that gives higher values when points are close
together and lower values when they're far apart.

## How It Works

The Exponential kernel is a radial basis function kernel that decreases exponentially with the
distance between data points. It is similar to the Gaussian kernel but uses the L1 norm (Manhattan distance)
instead of the L2 norm (Euclidean distance squared).

Think of the Exponential kernel as measuring similarity that fades quickly as points get farther apart.
It's like the brightness of a flashlight - very bright up close, but quickly gets dimmer as you move away.
Unlike some other kernels, the Exponential kernel never completely reaches zero, though it gets very close
for distant points.

This kernel is useful when you want a similarity measure that's sensitive to small distances but still
gives some weight to moderately distant points.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExponentialKernel()` | Initializes a new instance of the Exponential kernel with an optional scaling parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Exponential kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_sigma` | The scaling parameter that controls how quickly similarity decreases with distance. |

