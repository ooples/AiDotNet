---
title: "DeepSHAPExplanation<T>"
description: "Represents the result of a DeepSHAP analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a DeepSHAP analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attributions` | Gets or sets the feature attributions (SHAP values). |
| `ExpectedValue` | Gets or sets the expected (baseline) prediction value. |
| `FeatureNames` | Gets or sets the feature names. |
| `Instance` | Gets or sets the input instance. |
| `NumSamples` | Gets or sets the number of background samples used. |
| `OutputIndex` | Gets or sets the output index that was explained. |
| `Prediction` | Gets or sets the actual prediction for this instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedAttributions` | Gets attributions sorted by absolute value (most important first). |
| `GetSumError` | Verifies that attributions sum to (prediction - expected_value). |
| `ToString` | Returns a human-readable summary. |

