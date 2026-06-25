---
title: "JackknifeFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit using the jackknife resampling technique."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit using the jackknife resampling technique.

## For Beginners

This class helps determine if your machine learning model is a good fit for your data
using a technique called "jackknife resampling". This is like testing your model multiple times, 
each time leaving out one data point, to see how stable your model's performance is. 

If your model performs very differently when certain data points are removed, it might be overfitting
(memorizing the data rather than learning patterns). If it performs consistently regardless of which
points are removed, it's likely a more robust model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JackknifeFitDetector(JackknifeFitDetectorOptions)` | Initializes a new instance of the JackknifeFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates how confident the detector is in its assessment of the model fit. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes a model's performance data to determine the type of fit and provide recommendations. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines whether the model is overfitting, underfitting, or has a good fit. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations based on the detected fit type of the model. |
| `PerformJackknifeResampling(ModelEvaluationData<,,>)` | Performs jackknife resampling to evaluate model stability and calculate the average Mean Squared Error (MSE). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the jackknife fit detector. |

