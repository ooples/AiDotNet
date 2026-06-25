---
title: "SchedulerStepMode"
description: "Specifies when the learning rate scheduler should be stepped during training."
section: "API Reference"
---

`Enums` · `AiDotNet.LearningRateSchedulers`

Specifies when the learning rate scheduler should be stepped during training.

## For Beginners

This controls when the learning rate changes:

- Per batch: Changes after every mini-batch (more frequent, smoother changes)
- Per epoch: Changes after each complete pass through the dataset (most common)
- Warmup then epoch: Increases LR during warmup (per batch), then switches to per-epoch

## How It Works

Different training scenarios require different scheduling strategies. This enum
allows you to configure how frequently the learning rate is updated.

## Fields

| Field | Summary |
|:-----|:--------|
| `StepPerBatch` | Step the scheduler after each mini-batch. |
| `StepPerEpoch` | Step the scheduler after each epoch (default). |
| `WarmupThenEpoch` | Step per-batch during warmup phase, then switch to per-epoch. |

