---
title: "BootstrapFitDetector<T, TInput, TOutput>"
description: "A fit detector that uses bootstrap resampling to assess model fit and stability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that uses bootstrap resampling to assess model fit and stability.

## For Beginners

Bootstrap resampling is a statistical technique that creates multiple versions 
of your dataset by randomly sampling with replacement. This allows you to estimate the variability 
and stability of your model's performance metrics.

## How It Works

This detector uses bootstrap resampling to determine if a model is underfitting, overfitting, or has 
a good fit, while also assessing the confidence in this determination. Think of it like testing your 
model on many slightly different versions of your data to see how consistently it performs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BootstrapFitDetector(BootstrapFitDetectorOptions)` | Initializes a new instance of the BootstrapFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the bootstrap fit detection. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on bootstrap resampling. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on bootstrap resampling of model performance metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type. |
| `PerformBootstrap(ModelEvaluationData<,,>)` | Performs bootstrap resampling on the evaluation data. |
| `ResampleR2()` | Resamples an R² value by adding random noise. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the bootstrap fit detector. |
| `_random` | Random number generator used for bootstrap resampling. |

