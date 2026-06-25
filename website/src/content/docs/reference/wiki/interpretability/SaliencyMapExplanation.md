---
title: "SaliencyMapExplanation<T>"
description: "Represents the result of a Saliency Map analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a Saliency Map analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureNames` | Gets or sets the feature names. |
| `Input` | Gets or sets the input instance. |
| `Method` | Gets or sets the saliency method used. |
| `NormalizedSaliency` | Gets or sets the normalized saliency values (0 to 1). |
| `OutputIndex` | Gets or sets the output index that was explained. |
| `Prediction` | Gets or sets the model prediction. |
| `Saliency` | Gets or sets the raw saliency values. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedSaliency` | Gets saliency values sorted by absolute value (most salient first). |
| `GetTopSalientFeatures(Int32)` | Gets the top K most salient features. |
| `ToString` | Returns a human-readable summary. |

