---
title: "PruningStrategy"
description: "Strategy for pruning trials."
section: "API Reference"
---

`Enums` · `AiDotNet.HyperparameterOptimization`

Strategy for pruning trials.

## Fields

| Field | Summary |
|:-----|:--------|
| `MedianPruning` | Prune trials performing below the median at each step. |
| `PercentilePruning` | Prune trials in the bottom percentile at each step. |
| `SuccessiveHalving` | Successive halving: keep only the top half at each step. |
| `ThresholdPruning` | Prune based on explicit threshold (manual check required). |

