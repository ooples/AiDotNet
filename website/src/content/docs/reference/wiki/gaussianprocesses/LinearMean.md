---
title: "LinearMean<T>"
description: "Implements a linear mean function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a linear mean function.

## For Beginners

The linear mean function returns a linear combination of input features.

m(x) = w^T × x + b = w_1×x_1 + w_2×x_2 + ... + w_d×x_d + b

Where:

- w is the weight vector (one weight per input dimension)
- b is the bias (intercept)
- x is the input point

Why use linear mean?

- When you expect a linear trend in your data
- Combines well with kernels that capture nonlinear deviations
- Can be initialized from linear regression on your data

Example use cases:

- Time series with a linear trend
- Data where features have known linear effects
- When you want to separate trend from residuals

The GP will model deviations from this linear trend using the kernel.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearMean(Double[],Double)` | Initializes a linear mean function with double arrays. |
| `LinearMean(Int32)` | Initializes a zero-weight linear mean for the given number of dimensions. |
| `LinearMean(Vector<>,)` | Initializes a new linear mean function with specified weights and bias. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Bias` | Gets the bias term. |
| `NumDimensions` | Gets the number of dimensions. |
| `Weights` | Gets the weight vector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Matrix<>)` | Computes the linear mean values at multiple input points. |
| `Evaluate(Vector<>)` | Computes the linear mean value at the given input point. |
| `FromData(Matrix<>,Vector<>)` | Creates a linear mean function by fitting linear regression to training data. |
| `SolveLinearSystem(Matrix<>,Vector<>)` | Solves Ax = b using Gauss-Jordan elimination. |

