---
title: "ReduceOnPlateauScheduler"
description: "Reduces learning rate when a metric has stopped improving."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Reduces learning rate when a metric has stopped improving.

## For Beginners

Unlike other schedulers that follow a fixed schedule, this one
watches your model's performance and only reduces the learning rate when training gets "stuck"
(plateaus). If the model keeps improving, it keeps the learning rate the same. If improvement
stops for a while (patience epochs), it reduces the learning rate to allow finer adjustments.
Think of it like slowing down only when you notice you're not making progress.

## How It Works

ReduceOnPlateau monitors a quantity (usually validation loss) and reduces the learning rate
when no improvement is seen for a 'patience' number of evaluations. This is a reactive
scheduler that adapts based on training progress rather than a fixed schedule.

This scheduler requires you to call the Step(metric) overload with the monitored value.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReduceOnPlateauScheduler(Double,Double,Int32,Double,ThresholdMode,Int32,Mode,Double)` | Initializes a new instance of the ReduceOnPlateauScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestValue` | Gets the best metric value seen so far. |
| `Factor` | Gets the reduction factor. |
| `NumBadEpochs` | Gets the current number of bad epochs. |
| `Patience` | Gets the patience value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |
| `LoadState(Dictionary<String,Object>)` |  |
| `Reset` |  |
| `Step` |  |
| `Step(Double)` | Steps the scheduler with a metric value. |

