---
title: "TimeSeriesCrossValidationFitDetectorOptions"
description: "Configuration options for detecting overfitting, underfitting, and model stability in time series models using cross-validation techniques."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for detecting overfitting, underfitting, and model stability in time series models
using cross-validation techniques.

## For Beginners

This class helps you detect common problems when training time series forecasting models.

When building time series forecasting models:

- Overfitting: Model learns patterns specific to historical data that don't generalize to future data
- Underfitting: Model is too simple to capture important patterns in the data
- High variance: Model performance changes dramatically across different time periods

Time series cross-validation:

- Tests your model on multiple time periods
- Respects the temporal nature of the data (unlike regular cross-validation)
- Usually involves training on earlier data and testing on later data
- Helps assess how well your model will perform on future, unseen data

This class provides thresholds to automatically detect these issues based on
cross-validation results, helping you diagnose and fix model training problems.

## How It Works

Time series cross-validation is a technique for evaluating the performance and generalization ability of time 
series forecasting models. Unlike standard cross-validation used for non-time series data, time series 
cross-validation respects the temporal order of observations, typically using a rolling window or expanding 
window approach. This class provides configuration options for thresholds used to detect common modeling 
issues such as overfitting (where the model performs well on training data but poorly on validation data), 
underfitting (where the model performs poorly on both training and validation data), and high variance 
(where model performance varies significantly across different validation periods). These thresholds help 
automate the process of model evaluation and selection for time series forecasting tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `GoodFitThreshold` | Gets or sets the threshold for determining a good fit. |
| `HighVarianceThreshold` | Gets or sets the threshold for detecting high variance. |
| `OverfitThreshold` | Gets or sets the threshold for detecting overfitting. |
| `UnderfitThreshold` | Gets or sets the threshold for detecting underfitting. |

