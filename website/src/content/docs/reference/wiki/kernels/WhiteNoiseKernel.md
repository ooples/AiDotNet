---
title: "WhiteNoiseKernel<T>"
description: "Implements the White Noise kernel, which adds independent noise to each observation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the White Noise kernel, which adds independent noise to each observation.

## For Beginners

The White Noise kernel is a special kernel that models random measurement
noise in your data. It returns a non-zero value only when comparing a point to itself.

Think of it like this: every measurement you make has some random error. This error is
independent for each measurement - knowing the error of one measurement tells you nothing
about the error of another. The White Noise kernel captures this property.

In mathematical terms: k(x, x') = σ² if x = x', else 0

Where σ² is the noise variance (how much noise you expect in your measurements).

## How It Works

When to use the White Noise kernel:

- When your data has measurement noise that you want to model explicitly
- Combined with other kernels (like RBF + WhiteNoise) to separate signal from noise
- In Gaussian Process regression to account for observation noise

The White Noise kernel is rarely used alone - it's usually combined with other kernels
to create a more complete model of your data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WhiteNoiseKernel(Double,Double)` | Initializes a new instance of the White Noise kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the White Noise kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_noiseVariance` | The noise variance (σ²), which controls the magnitude of the noise. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_tolerance` | Tolerance for comparing vectors for equality. |

