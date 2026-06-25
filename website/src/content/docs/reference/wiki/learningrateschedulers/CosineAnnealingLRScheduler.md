---
title: "CosineAnnealingLRScheduler"
description: "Sets the learning rate using a cosine annealing schedule."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Sets the learning rate using a cosine annealing schedule.

## For Beginners

Instead of making sudden drops in learning rate, cosine annealing
provides a smooth, curved decrease that follows the shape of a cosine wave. The learning rate
starts high, decreases slowly at first, then more rapidly in the middle, and finally slows
down again as it approaches the minimum. This smooth transition often leads to better model
performance than abrupt changes.

## How It Works

CosineAnnealingLR uses a cosine function to smoothly decrease the learning rate from the
initial value to a minimum value over a specified number of steps. This is widely used
in modern deep learning and often outperforms step-based decay schedules.

Formula: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * step / T_max))

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosineAnnealingLRScheduler(Double,Int32,Double)` | Initializes a new instance of the CosineAnnealingLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EtaMin` | Gets the minimum learning rate. |
| `TMax` | Gets the maximum number of steps. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

