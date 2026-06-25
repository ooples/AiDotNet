---
title: "QueryStrategy"
description: "Strategy for selecting samples in active learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Quality`

Strategy for selecting samples in active learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `BALD` | Bayesian Active Learning by Disagreement (MC Dropout-based). |
| `LeastConfidence` | Select samples where the model's top prediction has lowest confidence. |
| `Margin` | Select samples with the smallest margin between top-2 predictions. |
| `Random` | Random sampling baseline. |
| `Uncertainty` | Select samples the model is most uncertain about (highest entropy). |

