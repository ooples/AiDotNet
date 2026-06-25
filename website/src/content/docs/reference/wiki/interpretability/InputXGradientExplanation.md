---
title: "InputXGradientExplanation<T>"
description: "Result of Input × Gradient attribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of Input × Gradient attribution.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attributions` | Gets or sets the attribution scores (input × gradient) for each feature. |
| `FeatureNames` | Gets or sets the feature names. |
| `Gradients` | Gets or sets the raw gradients for each feature. |
| `Instance` | Gets or sets the input instance. |
| `Prediction` | Gets or sets the model prediction. |
| `TargetClass` | Gets or sets the target class being explained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAttributionSum` | Gets the sum of attributions (for completeness check). |
| `GetSortedAttributions` | Gets attributions sorted by absolute magnitude. |
| `ToString` | Returns a human-readable summary. |

