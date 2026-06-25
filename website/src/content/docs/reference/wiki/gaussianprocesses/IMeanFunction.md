---
title: "IMeanFunction<T>"
description: "Interface for mean functions in Gaussian Processes."
section: "API Reference"
---

`Interfaces` · `AiDotNet.GaussianProcesses`

Interface for mean functions in Gaussian Processes.

## For Beginners

A mean function defines the "expected" value of the GP at any point
before we observe any data. It represents our prior belief about the function's behavior.

Common choices:

- Zero mean: We expect the function to hover around zero
- Constant mean: We expect the function to hover around some constant value
- Linear mean: We expect a linear trend in the data

The GP then models deviations from this mean using the kernel function.

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Matrix<>)` | Computes the mean values at multiple input points. |
| `Evaluate(Vector<>)` | Computes the mean value at a given input point. |

