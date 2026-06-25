---
title: "MissingValueStrategy"
description: "Strategies for handling missing values in robustness testing."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Options`

Strategies for handling missing values in robustness testing.

## Fields

| Field | Summary |
|:-----|:--------|
| `MeanImputation` | Replace with feature mean. |
| `MedianImputation` | Replace with feature median. |
| `ModeImputation` | Replace with feature mode (most common value). |
| `ModelNative` | Use model's native missing value handling. |
| `ZeroImputation` | Replace with zero. |

