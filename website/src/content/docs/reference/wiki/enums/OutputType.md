---
title: "OutputType"
description: "The expected format of model outputs that a loss function operates on."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

The expected format of model outputs that a loss function operates on.

## Fields

| Field | Summary |
|:-----|:--------|
| `Binary` | Binary values (0 or 1). |
| `Continuous` | Continuous real values (regression outputs). |
| `Distances` | Distance/similarity values. |
| `Logits` | Raw logits (unbounded real values, before Softmax). |
| `Probabilities` | Probability values in [0, 1] (after Softmax/Sigmoid). |

