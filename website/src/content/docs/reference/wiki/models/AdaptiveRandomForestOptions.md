---
title: "AdaptiveRandomForestOptions<T>"
description: "Configuration options for Adaptive Random Forest classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Adaptive Random Forest classifier.

## For Beginners

Adaptive Random Forest (ARF) is an ensemble method that combines
multiple Hoeffding trees with drift detection to handle evolving data streams.

## Properties

| Property | Summary |
|:-----|:--------|
| `DriftThreshold` | Gets or sets the drift threshold for tree replacement (DDM). |
| `GracePeriod` | Gets or sets the minimum samples before attempting splits. |
| `HoeffdingDelta` | Gets or sets the Hoeffding bound confidence parameter for individual trees. |
| `LambdaPoisson` | Gets or sets the lambda parameter for Poisson resampling. |
| `MaxTreeDepth` | Gets or sets the maximum depth for individual trees. |
| `NumBins` | Gets or sets the number of bins for numeric attribute discretization. |
| `NumFeaturesPerTree` | Gets or sets the number of features to consider for each tree. |
| `NumTrees` | Gets or sets the number of trees in the ensemble. |
| `TieThreshold` | Gets or sets the tie threshold for tree split decisions. |
| `WarningThreshold` | Gets or sets the warning threshold for drift detection (DDM). |

