---
title: "DifficultyEstimatorType"
description: "Types of difficulty estimators for curriculum learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Types of difficulty estimators for curriculum learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `ComplexityBased` | Uses sample complexity metrics (feature variance, etc.). |
| `ConfidenceBased` | Uses prediction confidence to estimate difficulty. |
| `Ensemble` | Combines multiple estimators for robust difficulty estimation. |
| `GradientBased` | Uses gradient magnitudes to estimate difficulty. |
| `LossBased` | Uses model loss on samples to estimate difficulty. |

