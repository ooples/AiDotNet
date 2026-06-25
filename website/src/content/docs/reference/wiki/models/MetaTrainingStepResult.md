---
title: "MetaTrainingStepResult<T>"
description: "Results from a single meta-training step (one outer loop update)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Results from a single meta-training step (one outer loop update).

## For Beginners

Think of this as the "score" for one training iteration.

Each meta-training iteration:

1. Samples a batch of tasks (e.g., 4 tasks)
2. Adapts to each task (inner loop)
3. Updates meta-parameters based on adaptation results (outer loop)
4. Returns these metrics to show how well that update performed

You'll get one of these for each iteration during training, allowing you to:

- Monitor training progress in real-time
- Log metrics to TensorBoard or similar tools
- Implement early stopping or learning rate schedules
- Debug training issues as they occur

## How It Works

This class represents metrics from one iteration of meta-training, tracking the
performance of a single outer loop update across a batch of tasks. It's a lightweight
snapshot designed for real-time monitoring during training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaTrainingStepResult(,,,Int32,Int32,Double,Dictionary<String,>)` | Initializes a new instance with metrics from one meta-training step. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Accuracy` | Gets the average accuracy across tasks in this step. |
| `AdditionalMetrics` | Gets algorithm-specific metrics for this training step. |
| `Iteration` | Gets the iteration number for this training step. |
| `MetaLoss` | Gets the meta-loss for this training step. |
| `NumTasks` | Gets the number of tasks processed in this meta-training step. |
| `TaskLoss` | Gets the average task-specific loss after adaptation. |
| `TimeMs` | Gets the time taken for this meta-training step in milliseconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a concise string representation for logging. |

