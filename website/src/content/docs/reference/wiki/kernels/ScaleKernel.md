---
title: "ScaleKernel<T>"
description: "A wrapper kernel that scales another kernel by a constant factor (output scale/variance)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

A wrapper kernel that scales another kernel by a constant factor (output scale/variance).

## For Beginners

The ScaleKernel is a simple but essential building block for GPs.
It wraps any base kernel and multiplies its output by a constant scale factor.

In mathematical terms:
k_scaled(x, x') = σ² × k_base(x, x')

Where:

- σ² is the output scale (variance) parameter
- k_base is the underlying kernel function

Why is this useful?

1. **Separates concerns**: The base kernel handles correlation structure,

while the scale controls the magnitude of variation

2. **Better optimization**: Having an explicit scale parameter often helps

hyperparameter optimization converge more easily

3. **Interpretability**: The scale directly relates to the variance of your output

Example: If your data varies between 0 and 100:

- Without scaling, the RBF kernel might not fit well
- With ScaleKernel(outputScale: 2500), the GP knows to expect variance of ~2500

## How It Works

Common usage patterns:

- ScaleKernel(rbfKernel, outputScale: variance_of_your_data)
- ScaleKernel(maternKernel, outputScale: 1.0) and optimize the scale

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScaleKernel(IKernelFunction<>,Double)` | Initializes a new ScaleKernel with the specified base kernel and output scale. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseKernel` | Gets the base kernel being scaled. |
| `OutputScale` | Gets or sets the output scale factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the scaled kernel value between two vectors. |
| `CalculateScaleGradient(Vector<>,Vector<>)` | Computes the gradient of the kernel with respect to the scale parameter. |
| `WithMatern(Double,Double,Double)` | Creates a ScaleKernel from a Matern kernel with specified parameters. |
| `WithRBF(Double,Double)` | Creates a ScaleKernel from an RBF (Gaussian) kernel with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseKernel` | The base kernel being scaled. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_outputScale` | The output scale factor (σ²). |

