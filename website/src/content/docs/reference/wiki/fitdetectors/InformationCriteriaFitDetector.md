---
title: "InformationCriteriaFitDetector<T, TInput, TOutput>"
description: "A detector that evaluates model fit using information criteria metrics (AIC and BIC)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitDetectors`

A detector that evaluates model fit using information criteria metrics (AIC and BIC).

## For Beginners

This class helps determine if your machine learning model is a good fit for your data
by using special metrics called "information criteria" (AIC and BIC). These metrics help balance
how well your model performs against how complex it is. A good model should explain your data well
without being unnecessarily complicated.

Think of it like this: if you're trying to draw a line through some points, you want a line that's
close to most points (good performance) but isn't so wiggly that it's just memorizing the exact
positions of each point (too complex).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InformationCriteriaFitDetector(InformationCriteriaFitDetectorOptions)` | Initializes a new instance of the InformationCriteriaFitDetector class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateConfidenceLevel(ModelEvaluationData<,,>)` | Calculates the confidence level in the fit determination. |
| `DetectFit(ModelEvaluationData<,,>)` | Analyzes model evaluation data to determine the type of fit and provide recommendations. |
| `DetermineFitType(ModelEvaluationData<,,>)` | Determines the type of fit based on information criteria metrics. |
| `GenerateRecommendations(FitType,ModelEvaluationData<,,>)` | Generates specific recommendations based on the detected fit type of the model. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_options` | Configuration options for the information criteria fit detector. |

