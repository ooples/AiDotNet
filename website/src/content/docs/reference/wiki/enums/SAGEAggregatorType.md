---
title: "SAGEAggregatorType"
description: "Aggregation function type for GraphSAGE."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Aggregation function type for GraphSAGE.

## For Beginners

These are different ways to combine information from neighbors.

- **Mean**: Average all neighbor features (balanced, smooth)
- **MaxPool**: Take the maximum value from neighbors (emphasizes outliers)
- **Sum**: Add up all neighbor features (sensitive to number of neighbors)

## Fields

| Field | Summary |
|:-----|:--------|
| `MaxPool` | Max pooling aggregation: takes maximum of neighbor features. |
| `Mean` | Mean aggregation: averages neighbor features. |
| `Sum` | Sum aggregation: sums neighbor features. |

