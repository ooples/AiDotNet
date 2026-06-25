---
title: "ConstantLRScheduler"
description: "Maintains a constant learning rate throughout training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Maintains a constant learning rate throughout training.

## For Beginners

This is the simplest scheduler - it just keeps the learning rate
the same throughout training. While adaptive schedules often work better, sometimes you want
a fixed learning rate, especially for fine-tuning or when the learning rate has already been
carefully tuned for your specific problem.

## How It Works

ConstantLR simply returns the same learning rate for every step. While this is the simplest
scheduler, it can be useful as a component in composite schedulers or for fine-tuning
where you want to keep the learning rate fixed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConstantLRScheduler(Double)` | Initializes a new instance of the ConstantLRScheduler class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |

