---
title: "BayesianFitDetector<T, TInput, TOutput>"
description: "A fit detector that uses Bayesian model comparison metrics to assess model fit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that uses Bayesian model comparison metrics to assess model fit.

## For Beginners

Bayesian statistics provides a framework for model evaluation that considers 
both how well a model fits the data and its complexity. This detector uses several Bayesian metrics 
to determine if a model is underfitting, overfitting, or has a good fit.

## How It Works

Unlike traditional methods that only look at prediction errors, Bayesian methods also consider 
the model's complexity and uncertainty, providing a more comprehensive assessment of model fit.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BayesianFitDetector(BayesianFitDetectorOptions)` | Initializes a new instance of the BayesianFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the Bayesian fit detection. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on Bayesian model comparison metrics. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on Bayesian model comparison metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected fit type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the Bayesian fit detector. |

