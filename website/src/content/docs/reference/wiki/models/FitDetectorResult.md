---
title: "FitDetectorResult<T>"
description: "Represents the result of a model fit detection analysis, which evaluates how well a model fits the data and provides recommendations for improvement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the result of a model fit detection analysis, which evaluates how well a model fits the data
and provides recommendations for improvement.

## For Beginners

This class helps you understand how well your model fits your data.

When building statistical or machine learning models:

- You need to know if your model is a good match for your data
- Models can underfit (too simple) or overfit (too complex)
- Different types of models have different fit characteristics

This class stores:

- What type of fit was detected (good, underfit, overfit, etc.)
- How confident the detector is in its assessment
- Specific recommendations to improve your model
- Additional information that might be useful for diagnosis

This information helps you make informed decisions about how to adjust your model
to achieve better performance on both training and new data.

## How It Works

When building statistical or machine learning models, it's important to assess how well the model fits the data. 
This class stores the results of such an assessment, including the type of fit detected (e.g., good fit, 
underfitting, overfitting), a confidence level for that assessment, and recommendations for improving the model. 
It also provides a flexible dictionary for storing additional information specific to different types of models 
or detection algorithms. This information helps data scientists and developers understand the quality of their 
models and take appropriate actions to improve them.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FitDetectorResult` | Initializes a new instance of the FitDetectorResult class with an empty list of recommendations. |
| `FitDetectorResult(FitType,)` | Initializes a new instance of the FitDetectorResult class with the specified fit type and confidence level. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdditionalInfo` | Gets or sets additional information about the fit detection result. |
| `ConfidenceLevel` | Gets or sets the confidence level for the fit type assessment. |
| `FitType` | Gets or sets the type of fit detected for the model. |
| `Recommendations` | Gets or sets a list of recommendations for improving the model fit. |

