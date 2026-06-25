---
title: "SequentialLRScheduler"
description: "Chains multiple learning rate schedulers together in sequence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Chains multiple learning rate schedulers together in sequence.

## For Beginners

Sometimes you want different learning rate strategies at
different points in training. For example, you might want linear warmup for the first
1000 steps, then cosine annealing for the next 9000 steps. This scheduler lets you
chain multiple schedulers together, specifying when to switch from one to the next.

## How It Works

SequentialLR allows you to compose multiple schedulers, each running for a specified
number of steps. This is useful for complex training schedules that combine different
strategies at different phases of training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequentialLRScheduler(IList<ILearningRateScheduler>,Int32[])` | Initializes a new instance of the SequentialLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentScheduler` | Gets the current active scheduler. |
| `CurrentSchedulerIndex` | Gets the current active scheduler index. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |
| `LoadState(Dictionary<String,Object>)` |  |
| `Reset` |  |
| `Step` |  |

