---
title: "PredictionAggregationMethod"
description: "Specifies how to combine predictions from multiple augmented versions of the same input."
section: "API Reference"
---

`Enums` · `AiDotNet.Augmentation`

Specifies how to combine predictions from multiple augmented versions of the same input.

## For Beginners

When you make predictions on several variations of an image
(flipped, rotated, etc.), you need a way to combine those predictions into one final answer.
This enum controls how that combination happens.

Think of it like asking 5 friends to estimate the price of a used car:

- **Mean:** Take the average of all estimates ($15K + $18K + $16K + $14K + $17K) / 5 = $16K
- **Median:** Take the middle value when sorted ($14K, $15K, [$16K], $17K, $18K) = $16K
- **Vote:** If 3 friends say "buy" and 2 say "don't buy", go with "buy"

## Fields

| Field | Summary |
|:-----|:--------|
| `GeometricMean` | Multiply predictions together (then take the Nth root). |
| `Max` | Take the highest prediction. |
| `Mean` | Average all predictions together. |
| `Median` | Take the middle prediction when sorted. |
| `Min` | Take the lowest prediction. |
| `Vote` | Count votes from each prediction and pick the winner. |
| `WeightedMean` | Like Mean, but gives more weight to higher-confidence predictions. |

