---
title: "IntegratedGradientsExplanation<T>"
description: "Represents the result of an Integrated Gradients analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of an Integrated Gradients analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attributions` | Gets or sets the feature attributions. |
| `Baseline` | Gets or sets the baseline used. |
| `BaselinePrediction` | Gets or sets the prediction at the baseline. |
| `ConvergenceDelta` | Gets or sets the convergence delta (difference between sum of attributions and prediction difference). |
| `FeatureNames` | Gets or sets the feature names. |
| `Input` | Gets or sets the input instance. |
| `InputPrediction` | Gets or sets the prediction at the input. |
| `NumSteps` | Gets or sets the number of integration steps used. |
| `OutputIndex` | Gets or sets the output index that was explained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSortedAttributions` | Gets attributions sorted by absolute value (most important first). |
| `ToString` | Returns a human-readable summary. |

