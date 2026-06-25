---
title: "SGDLossType"
description: "Loss function types for SGD regression."
section: "API Reference"
---

`Enums` · `AiDotNet.OnlineLearning`

Loss function types for SGD regression.

## Fields

| Field | Summary |
|:-----|:--------|
| `EpsilonInsensitive` | Epsilon-insensitive loss: Zero for errors smaller than epsilon. |
| `Huber` | Huber loss: Squared for small errors, linear for large errors. |
| `SquaredError` | Squared error loss: (y - ŷ)². |

