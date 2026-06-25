---
title: "SVMBase<T>"
description: "Base class for Support Vector Machine classifiers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Classification.SVM`

Base class for Support Vector Machine classifiers.

## For Beginners

SVMs are like finding the best possible line (or curve) to separate different groups.
Unlike other methods that just find "a" line that works, SVMs find "the best" line
by maximizing the gap (margin) between the line and the nearest points from each class.

Key SVM concepts:

- Margin: The gap between the decision boundary and the nearest training points
- Support Vectors: The training points closest to the decision boundary
- Kernel Trick: A way to handle non-linear boundaries without explicitly computing new features

## How It Works

Support Vector Machines (SVMs) are powerful classifiers that find the optimal
hyperplane separating classes with maximum margin. This base class provides
common functionality for SVM implementations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SVMBase(SVMOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the SVMBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NSupportVectors` |  |
| `Options` | Gets the SVM specific options. |
| `SupportVectors` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `ComputeKernel(Vector<>,Vector<>)` | Computes the kernel between two vectors. |
| `ComputeLaplacianKernel(Vector<>,Vector<>)` | Computes Laplacian kernel: K(x, y) = exp(-gamma * \|\|x - y\|\|_1) |
| `ComputeLinearKernel(Vector<>,Vector<>)` | Computes linear kernel: K(x, y) = x · y |
| `ComputePolynomialKernel(Vector<>,Vector<>)` | Computes polynomial kernel: K(x, y) = (gamma * x · y + coef0)^degree |
| `ComputeRBFKernel(Vector<>,Vector<>)` | Computes RBF kernel: K(x, y) = exp(-gamma * \|\|x - y\|\|^2) |
| `ComputeRBFKernelArrays([],[])` | Array-typed RBF kernel for callers that already have backing arrays (e.g. |
| `ComputeSigmoidKernel(Vector<>,Vector<>)` | Computes sigmoid kernel: K(x, y) = tanh(gamma * x · y + coef0) |
| `DecisionFunction(Matrix<>)` |  |
| `GetGamma` | Gets the gamma value, computing it automatically if not specified. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `GetRow(Matrix<>,Int32)` | Extracts a row from a matrix as a vector. |
| `GetRowArray(Matrix<>,Int32)` | Same as `Int32)` but returns a raw array — avoids the `Vector` indexer's deferred-materializer monitor on hot paths. |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dualCoef` | The dual coefficients for the support vectors. |
| `_intercept` | The bias terms for each classifier. |
| `_supportVectors` | The support vectors learned during training. |

