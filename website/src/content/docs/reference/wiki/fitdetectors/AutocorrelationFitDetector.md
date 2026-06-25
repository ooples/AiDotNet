---
title: "AutocorrelationFitDetector<T, TInput, TOutput>"
description: "A fit detector that analyzes autocorrelation in model residuals to assess model fit."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A fit detector that analyzes autocorrelation in model residuals to assess model fit.

## For Beginners

Autocorrelation refers to the correlation of a time series with its own past values. 
In the context of model fitting, autocorrelation in residuals (prediction errors) can indicate that the 
model is missing important patterns in the data.

## How It Works

This detector uses the Durbin-Watson statistic to measure autocorrelation in model residuals. The 
Durbin-Watson statistic ranges from 0 to 4, with a value of 2 indicating no autocorrelation, values 
less than 2 indicating positive autocorrelation, and values greater than 2 indicating negative 
autocorrelation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutocorrelationFitDetector(AutocorrelationFitDetectorOptions)` | Initializes a new instance of the AutocorrelationFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level of the autocorrelation detection. |
| `DetectFit(ModelEvaluationData<,,>)` | Detects the fit type of a model based on autocorrelation in residuals. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the fit type based on the Durbin-Watson statistic of model residuals. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates recommendations based on the detected autocorrelation type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the autocorrelation fit detector. |

