---
title: "TimeSeriesForestOptions<T>"
description: "Configuration options for the Time Series Forest classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Time Series Forest classifier.

## For Beginners

Time Series Forest is an ensemble method that builds decision trees
on randomly selected intervals of the time series. Each tree looks at a different portion
of the sequence, and the ensemble combines their predictions for robust classification.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDepth` | Gets or sets the maximum depth of each tree. |
| `MaxIntervalFraction` | Gets or sets the maximum interval length as a fraction of sequence length. |
| `MinIntervalFraction` | Gets or sets the minimum interval length as a fraction of sequence length. |
| `MinSamplesSplit` | Gets or sets the minimum number of samples required to split a node. |
| `NumTrees` | Gets or sets the number of trees in the forest. |
| `RandomSeed` | Gets or sets the random seed for reproducible results. |

