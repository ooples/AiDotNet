---
title: "PNAAggregator"
description: "Aggregation function types for Principal Neighbourhood Aggregation (PNA)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Aggregation function types for Principal Neighbourhood Aggregation (PNA).

## For Beginners

These are different ways to combine information from neighbor nodes:

- **Mean**: Average all neighbor features (balanced, smooth)
- **Max**: Take the maximum value (emphasizes strong signals)
- **Min**: Take the minimum value (emphasizes weak signals)
- **Sum**: Add up all features (sensitive to number of neighbors)
- **StdDev**: Standard deviation (captures variance in neighborhood)

## Fields

| Field | Summary |
|:-----|:--------|
| `Max` | Max aggregation - takes maximum of neighbor features. |
| `Mean` | Mean aggregation - averages neighbor features. |
| `Min` | Min aggregation - takes minimum of neighbor features. |
| `StdDev` | Standard deviation aggregation - computes std of neighbor features. |
| `Sum` | Sum aggregation - sums neighbor features. |

