---
title: "LinearWarmupScheduler"
description: "Implements linear learning rate warmup followed by constant or decay schedule."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Implements linear learning rate warmup followed by constant or decay schedule.

## For Beginners

When training starts, the model's weights are random and
can produce large, unstable gradients. Starting with a very small learning rate and
gradually increasing it (warmup) helps the model stabilize before moving to the full
learning rate. Think of it like warming up an engine before driving at full speed.

## How It Works

Linear warmup gradually increases the learning rate from a small initial value to the
target learning rate over a specified number of warmup steps. This is commonly used
in transformer training and helps stabilize early training dynamics.

This scheduler supports three modes after warmup:

- Constant: Keep the base learning rate after warmup
- Linear decay: Linearly decrease to a minimum value
- Cosine decay: Use cosine annealing to decrease to a minimum value

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearWarmupScheduler(Double,Int32,Int32,Double,Nullable<DecayMode>,Double)` | Initializes a new instance of the LinearWarmupScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentDecayMode` | Gets the decay mode. |
| `TotalSteps` | Gets the total number of steps. |
| `WarmupSteps` | Gets the number of warmup steps. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

