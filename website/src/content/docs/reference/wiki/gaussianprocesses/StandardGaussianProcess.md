---
title: "StandardGaussianProcess<T>"
description: "Implements a standard Gaussian Process regression model for making probabilistic predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a standard Gaussian Process regression model for making probabilistic predictions.

## For Beginners

A Gaussian Process is a flexible machine learning method that can make predictions
with uncertainty estimates.

Think of it like drawing a line through data points, but instead of just one line, it gives you
a range of possible lines with a confidence level for each. This helps you understand not just
what the prediction is, but how certain the model is about that prediction.

Gaussian Processes are particularly useful when:

- You have a small to medium amount of data
- You need to know how confident the model is in its predictions
- Your data might have complex patterns that simpler models can't capture

Unlike many other machine learning methods, Gaussian Processes don't just learn a fixed set of
parameters - they use all training data when making predictions, which allows them to capture
complex patterns in your data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StandardGaussianProcess(IKernelFunction<>,MatrixDecompositionType,Double)` | Initializes a new instance of the StandardGaussianProcess class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddJitter(Matrix<>)` | Calculates the kernel values between a set of data points and a single point. |
| `CalculateKernelMatrix(Matrix<>,Matrix<>)` | Calculates the kernel matrix between two sets of data points. |
| `Fit(Matrix<>,Vector<>)` | Trains the Gaussian Process model on the provided data. |
| `Predict(Vector<>)` | Makes a prediction for a new data point, returning both the predicted value and its uncertainty. |
| `SolveWithFallback(Matrix<>,Vector<>)` | Solves a linear system using the configured decomposition type, falling back to SVD if the result contains NaN. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel function used by the model and recalculates the kernel matrix if training data exists. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_K` | The kernel matrix calculated from the training data. |
| `_X` | The matrix of input features from the training data. |
| `_decompositionType` | The method used to decompose matrices for solving linear systems. |
| `_kernel` | The kernel function that determines how similarity between data points is calculated. |
| `_noiseVariance` | Observation noise variance σ²_n. |
| `_numOps` | Operations for performing numeric calculations with the generic type T. |
| `_y` | The vector of target values from the training data. |

