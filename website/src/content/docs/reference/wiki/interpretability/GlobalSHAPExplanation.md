---
title: "GlobalSHAPExplanation<T>"
description: "Represents global SHAP explanations aggregated across multiple instances."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents global SHAP explanations aggregated across multiple instances.

## For Beginners

While local SHAP explains one prediction, global SHAP
helps you understand which features are important across ALL predictions.

It aggregates individual explanations to show:

- Which features have the biggest impact on average
- How consistent that impact is across different predictions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlobalSHAPExplanation(SHAPExplanation<>[],String[])` | Initializes a global SHAP explanation from local explanations. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureNames` | Gets the feature names, if available. |
| `LocalExplanations` | Gets the individual explanations. |
| `NumFeatures` | Gets the number of features. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFeatureImportance` | Gets feature importance ranking based on mean absolute SHAP values. |
| `GetMeanAbsoluteShapValues` | Gets the mean absolute SHAP value for each feature (global feature importance). |
| `GetMeanShapValues` | Gets the mean SHAP value for each feature (can be positive or negative). |
| `GetShapValueStdDev` | Gets the standard deviation of SHAP values for each feature. |

