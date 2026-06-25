---
title: "LearningRateSchedule"
description: "Learning rate schedule types for online learning."
section: "API Reference"
---

`Enums` · `AiDotNet.OnlineLearning`

Learning rate schedule types for online learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `Constant` | Constant learning rate throughout training. |
| `Exponential` | Learning rate decreases exponentially. |
| `InverseScaling` | Learning rate decreases as 1/t (inverse time scaling). |
| `StepDecay` | Learning rate drops by half every N samples. |

