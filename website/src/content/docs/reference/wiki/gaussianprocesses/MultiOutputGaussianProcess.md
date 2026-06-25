---
title: "MultiOutputGaussianProcess<T>"
description: "A Gaussian Process model that can predict multiple output values simultaneously."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

A Gaussian Process model that can predict multiple output values simultaneously.

## For Beginners

A Gaussian Process is a flexible machine learning method that can learn patterns from data
and provide uncertainty estimates with its predictions. Think of it as drawing a smooth curve through your data points,
but also showing how confident it is about different parts of that curve.

This "Multi-Output" version can predict multiple related values at once. For example, if you're predicting
the temperature and humidity for weather forecasting, this model can learn how these outputs relate to each other
and make better predictions by considering them together.

Unlike simpler models that just give you a single prediction, Gaussian Processes also tell you how confident
they are about each prediction (the "variance" or "uncertainty"). This is especially useful when making decisions
based on predictions where knowing the confidence level is important.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiOutputGaussianProcess(IKernelFunction<>)` | Creates a new instance of the MultiOutputGaussianProcess with the specified kernel function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddJitter(Matrix<>)` | Calculates the kernel matrix between two sets of input points. |
| `CalculateKernelVector(Matrix<>,Vector<>)` | Calculates the kernel vector between a set of input points and a single input point. |
| `Fit(Matrix<>,Vector<>)` | This method is not supported for multi-output Gaussian Processes. |
| `FitMultiOutput(Matrix<>,Matrix<>)` | Trains the Gaussian Process model on the provided multi-output training data. |
| `Predict(Vector<>)` | This method is not supported for multi-output Gaussian Processes. |
| `PredictMultiOutput(Vector<>)` | Makes predictions for a new input point, returning both the predicted means and the covariance matrix. |
| `UpdateKernel(IKernelFunction<>)` | Updates the kernel function used by the Gaussian Process and retrains the model if data is available. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_K` | The kernel matrix calculated from the training data. |
| `_L` | The Cholesky decomposition of the kernel matrix, used for efficient calculations. |
| `_X` | The input training data matrix. |
| `_Y` | The output training data matrix (multiple outputs). |
| `_alpha` | The alpha matrix used for making predictions. |
| `_kernel` | The kernel function that determines how points in the input space relate to each other. |
| `_numOps` | Operations for the numeric type T. |

