---
title: "TreeSHAPExplanation<T>"
description: "Represents the result of a TreeSHAP analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a TreeSHAP analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedValue` | Gets or sets the expected (baseline) prediction value. |
| `FeatureNames` | Gets or sets the feature names. |
| `Instance` | Gets or sets the input instance. |
| `Prediction` | Gets or sets the actual prediction for this instance. |
| `ShapValues` | Gets or sets the SHAP values for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNegativeContributions` | Gets features that pushed the prediction lower. |
| `GetPositiveContributions` | Gets features that pushed the prediction higher. |
| `GetSortedAttributions` | Gets attributions sorted by absolute value (most important first). |
| `GetSumError` | Verifies that SHAP values sum to (prediction - expected_value). |
| `ToString` | Returns a human-readable summary. |

