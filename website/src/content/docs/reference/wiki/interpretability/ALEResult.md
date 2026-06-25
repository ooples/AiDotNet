---
title: "ALEResult<T>"
description: "Represents the result of an ALE (Accumulated Local Effects) analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of an ALE (Accumulated Local Effects) analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `ALEValues` | Gets or sets the ALE values for each feature (feature index -> ALE values at interval boundaries). |
| `FeatureIndices` | Gets or sets the feature indices analyzed. |
| `FeatureNames` | Gets or sets the feature names. |
| `IntervalBounds` | Gets or sets the interval bounds for each feature (feature index -> bounds). |
| `IntervalCounts` | Gets or sets the count of data points in each interval (feature index -> counts). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary. |

