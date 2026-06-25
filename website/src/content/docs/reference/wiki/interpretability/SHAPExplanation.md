---
title: "SHAPExplanation<T>"
description: "Represents a SHAP explanation for a single prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents a SHAP explanation for a single prediction.

## For Beginners

A SHAP explanation tells you exactly how much each feature
contributed to a specific prediction.

Key concepts:

- **Baseline Value**: The average prediction (what the model predicts "by default")
- **SHAP Values**: How much each feature pushed the prediction up or down from the baseline
- **Prediction**: Should equal Baseline + Sum(SHAP Values)

Example: If predicting house prices:

- Baseline: $300,000 (average house price)
- Bedrooms: +$50,000 (having 4 bedrooms adds value)
- Location: +$100,000 (good neighborhood)
- Age: -$30,000 (older house reduces value)
- Prediction: $420,000

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SHAPExplanation(Vector<>,,,String[])` | Initializes a new SHAP explanation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaselineValue` | Gets the baseline (expected) prediction value. |
| `FeatureNames` | Gets the feature names, if available. |
| `NumFeatures` | Gets the number of features. |
| `Prediction` | Gets the actual prediction for this instance. |
| `ShapValues` | Gets the SHAP values for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetShapValue(Int32)` | Gets the SHAP value for a specific feature by index. |
| `GetShapValue(String)` | Gets the SHAP value for a specific feature by name. |
| `GetSortedFeatures` | Gets features sorted by absolute SHAP value (most important first). |
| `GetTopFeatures(Int32)` | Gets the top N most important features by absolute SHAP value. |
| `ToString` | Returns a human-readable summary of the explanation. |
| `VerifyConsistency(Double)` | Verifies that SHAP values sum to the difference between prediction and baseline. |

