---
title: "MetaSGDLearningRateScheduleType"
description: "Learning rate schedule types for Meta-SGD meta-training."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Learning rate schedule types for Meta-SGD meta-training.

## How It Works

These schedules control how the meta-learning rate (outer loop) changes
during meta-training.

## Fields

| Field | Summary |
|:-----|:--------|
| `Constant` | No scheduling, constant learning rate throughout training. |
| `CosineAnnealing` | Cosine annealing schedule for smooth decay. |
| `Cyclical` | Cyclical learning rates that oscillate between bounds. |
| `Exponential` | Exponential decay over time. |
| `StepDecay` | Step decay at specified intervals (e.g., halve every N episodes). |

