---
title: "GaussianProcessFitDetector<T, TInput, TOutput>"
description: "A fit detector that uses Gaussian Process regression to analyze model uncertainty and performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that uses Gaussian Process regression to analyze model uncertainty and performance.

## For Beginners

Gaussian Process regression is a probabilistic machine learning technique that 
not only makes predictions but also provides uncertainty estimates for those predictions. This detector 
uses these uncertainty estimates to assess how well a model fits the data.

## How It Works

By analyzing both the accuracy of predictions (RMSE) and the uncertainty in those predictions, this 
detector can identify issues like overfitting (high confidence but poor performance) or underfitting 
(high uncertainty and poor performance).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianProcessFitDetector(GaussianProcessFitDetectorOptions)` | Initializes a new instance of the GaussianProcessFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the Gaussian Process-based fit detection. |
| `CalculateKernelMatrix(Matrix<>,Matrix<>)` | Calculates the kernel matrix between two sets of points. |
| `CalculateLogLikelihood(Matrix<>,Vector<>)` | Calculates the log likelihood of the Gaussian Process model. |
| `CalculateRBFKernel(Vector<>,Vector<>)` | Calculates the RBF (Radial Basis Function) kernel between two points. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on Gaussian Process regression analysis. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on Gaussian Process regression analysis. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type and Gaussian Process analysis. |
| `OptimizeHyperparameters(Matrix<>,Vector<>)` | Optimizes the hyperparameters for the Gaussian Process model. |
| `PerformGaussianProcessRegression(ModelEvaluationData<,,>)` | Performs Gaussian Process regression on the model's residuals. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Gaussian Process fit detector. |

