---
title: "PartialDependenceResult<T>"
description: "Represents the result of a partial dependence analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a partial dependence analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureIndices` | Gets or sets the feature indices analyzed. |
| `FeatureNames` | Gets or sets the feature names. |
| `GridResolution` | Gets the grid resolution. |
| `GridValues` | Gets or sets the grid values for each feature (feature index -> grid values). |
| `IceCurves` | Gets or sets the ICE curves for each feature (feature index -> [sample, grid point]). |
| `PartialDependence` | Gets or sets the partial dependence values for each feature (feature index -> PD values). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary. |

