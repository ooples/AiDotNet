---
title: "EarlyStoppingMode"
description: "Mode for determining improvement in early stopping."
section: "API Reference"
---

`Enums` · `AiDotNet.HyperparameterOptimization`

Mode for determining improvement in early stopping.

## Fields

| Field | Summary |
|:-----|:--------|
| `Best` | Compare against the best value seen so far. |
| `MovingAverage` | Compare against a moving average of recent values. |
| `RelativeBest` | Compare using relative improvement (percentage-based). |

