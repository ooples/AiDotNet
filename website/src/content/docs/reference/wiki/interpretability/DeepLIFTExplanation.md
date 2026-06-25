---
title: "DeepLIFTExplanation<T>"
description: "Represents the result of a DeepLIFT analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a DeepLIFT analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attributions` | Gets or sets the feature attributions. |
| `Baseline` | Gets or sets the baseline used. |
| `BaselinePrediction` | Gets or sets the prediction at the baseline. |
| `CompletenessError` | Gets or sets the completeness error (difference between sum of attributions and delta output). |
| `DeltaOutput` | Gets or sets the difference between input and baseline predictions. |
| `FeatureNames` | Gets or sets the feature names. |
| `Input` | Gets or sets the input instance. |
| `InputPrediction` | Gets or sets the prediction at the input. |
| `OutputIndex` | Gets or sets the output index that was explained. |
| `Rule` | Gets or sets the DeepLIFT rule used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNegativeContributions` | Gets negative contributions (features that decreased prediction). |
| `GetPositiveContributions` | Gets positive contributions (features that increased prediction). |
| `GetSortedAttributions` | Gets attributions sorted by absolute value (most important first). |
| `ToString` | Returns a human-readable summary. |

