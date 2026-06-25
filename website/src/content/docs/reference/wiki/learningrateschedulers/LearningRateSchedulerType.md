---
title: "LearningRateSchedulerType"
description: "Enumeration of available learning rate scheduler types."
section: "API Reference"
---

`Enums` · `AiDotNet.LearningRateSchedulers`

Enumeration of available learning rate scheduler types.

## How It Works

Use this enum with the `LearningRateSchedulerFactory` to create
schedulers by type without having to reference the concrete classes directly.

## Fields

| Field | Summary |
|:-----|:--------|
| `Constant` | Constant learning rate (no decay). |
| `CosineAnnealing` | Cosine annealing: smooth cosine-shaped decay. |
| `CosineAnnealingWarmRestarts` | Cosine annealing with warm restarts (SGDR). |
| `Cyclic` | Cyclic learning rate: oscillate between bounds. |
| `Exponential` | Exponential decay: multiply LR by gamma every epoch. |
| `Lambda` | Custom lambda function scheduler. |
| `LinearWarmup` | Linear warmup followed by optional decay. |
| `MultiStep` | Multi-step decay: multiply LR by gamma at specified milestones. |
| `OneCycle` | One cycle policy: warmup then annealing. |
| `Polynomial` | Polynomial decay: LR follows polynomial curve to end value. |
| `ReduceOnPlateau` | Reduce on plateau: decrease when metric stops improving. |
| `Sequential` | Sequential composition of multiple schedulers. |
| `Step` | Step decay: multiply LR by gamma every step_size epochs. |

