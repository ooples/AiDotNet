---
title: "ResidualBootstrapFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit quality using residual bootstrap resampling techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit quality using residual bootstrap resampling techniques.

## For Beginners

This class helps determine if your machine learning model is a good fit for your data.
It uses a technique called "bootstrap resampling" which creates many simulated datasets by
randomly reusing the errors (residuals) from your original model. This helps understand if your
model is too complex (overfit), too simple (underfit), or just right (good fit).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResidualBootstrapFitDetector(ResidualBootstrapFitDetectorOptions)` | Initializes a new instance of the ResidualBootstrapFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates how confident the detector is in its assessment of the model's fit type. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model evaluation data to determine the fit type and provide recommendations. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit (overfit, underfit, or good fit) based on bootstrap analysis. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates practical recommendations for improving the model based on the detected fit type. |
| `PerformResidualBootstrap(ModelEvaluationData<,,>)` | Performs residual bootstrap resampling to generate multiple simulated datasets and calculate their error metrics. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the residual bootstrap fit detector. |
| `_random` | Random number generator used for bootstrap resampling. |

