---
title: "GradientSHAPExplanation<T>"
description: "Represents the result of a GradientSHAP analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a GradientSHAP analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttributionVariances` | Gets or sets the variance of attributions across samples. |
| `Attributions` | Gets or sets the SHAP values (feature attributions). |
| `ExpectedValue` | Gets or sets the expected value (average prediction over background data). |
| `FeatureNames` | Gets or sets the feature names. |
| `InputPrediction` | Gets or sets the prediction for the input. |
| `NumSamples` | Gets or sets the number of samples used. |
| `OutputIndex` | Gets or sets the output index that was explained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAttributionsWithConfidence(Double)` | Gets confidence intervals for attributions. |
| `GetSortedAttributions` | Gets attributions sorted by absolute value (most important first). |
| `ToString` | Returns a human-readable summary. |

