---
title: "SparseGaussianProcess<T>"
description: "A sparse implementation of Gaussian Process regression that uses inducing points to reduce computational complexity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

A sparse implementation of Gaussian Process regression that uses inducing points to reduce computational complexity.

## For Beginners

A Gaussian Process is a flexible machine learning method that can model complex relationships in data.

The "sparse" version solves a common problem with Gaussian Processes - they can be very slow with large datasets.
Instead of using all data points for predictions (which can be computationally expensive), 
this implementation selects a smaller set of representative points called "inducing points" 
that capture the essential patterns in your data.

Think of it like summarizing a book: instead of reading every word (standard GP),
you read just the chapter summaries (sparse GP) to get the main ideas more efficiently.

This approach makes Gaussian Processes practical for larger datasets while maintaining most of their predictive power.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateKernelDiagonal(Matrix<>)` | Calculates the kernel values of each data point with itself. |
| `CalculateKernelMatrix(Matrix<>,Matrix<>)` | Calculates the kernel matrix between two sets of data points. |
| `CalculateKernelVector(Matrix<>,Vector<>)` | Calculates the kernel values between a set of data points and a single point. |
| `Fit(Matrix<>,Vector<>)` | Trains the Gaussian Process model on the provided data. |
| `IsAllFinite(Vector<>)` | Returns true only if every element of `v` is a finite number (no NaN, no infinity). |
| `Predict(Vector<>)` | Makes a prediction for a new input point, returning both the mean prediction and its variance. |
| `Reciprocal()` | Calculates the reciprocal (1/x) of a value. |
| `SelectInducingPoints(Matrix<>)` | Selects a subset of data points to use as inducing points for the sparse Gaussian Process. |
| `SolveViaPseudoInverse(Matrix<>,Vector<>)` | Solves A·x = b via the Moore–Penrose pseudoinverse computed from a truncated SVD, dropping singular values below `ε · σ_max`. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel function used by the model and retrains if data is available. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_D` | Diagonal elements used in the sparse approximation. |
| `_Kuu` | The kernel matrix of inducing points (with jitter), used for prediction. |
| `_L` | The lower triangular matrix from Cholesky decomposition of the kernel matrix. |
| `_V` | Intermediate matrix used for efficient predictions. |
| `_X` | The training input data matrix. |
| `_alpha` | Weights vector used for mean prediction. |
| `_decompositionType` | The type of matrix decomposition to use for solving linear systems. |
| `_inducingPoints` | A subset of training points used to approximate the full Gaussian Process. |
| `_kernel` | The kernel function that defines the similarity between data points. |
| `_noiseVariance` | Initializes a new instance of the SparseGaussianProcess class. |
| `_numOps` | Helper for performing numeric operations on type T. |
| `_y` | The training output data vector. |

