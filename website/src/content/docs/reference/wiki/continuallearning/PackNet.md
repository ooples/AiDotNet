---
title: "PackNet<T>"
description: "Implements PackNet for continual learning through parameter isolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements PackNet for continual learning through parameter isolation.

## For Beginners

PackNet achieves continual learning by dynamically pruning
and freezing network weights. After learning each task, unimportant weights are pruned,
and the remaining weights are frozen. New tasks can only use the pruned (free) weights,
effectively isolating each task's parameters.

## How It Works

**How it works:**

**Key Concepts:**

**Advantages:**

**Reference:** Mallya, A. and Lazebnik, S. "PackNet: Adding Multiple Tasks to a
Single Network by Iterative Pruning" (2018). CVPR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PackNet(Double,Double)` | Initializes a new instance of the PackNet class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `FreeWeightCount` | Gets the number of free weights available for new tasks. |
| `PruningRatio` | Gets the pruning ratio. |
| `TaskCount` | Gets the number of tasks stored. |
| `TotalParameters` | Gets the total parameter count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `ApplyTaskMask(INeuralNetwork<>,Int32)` | Applies the task-specific mask to network parameters for inference. |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `GetAllWeightsUpToTask(Int32)` | Gets all weights used by tasks up to and including the specified task. |
| `GetTaskMask(Int32)` | Gets the weight mask for a specific task. |
| `GetWeightAllocationStats` | Gets statistics about weight allocation across tasks. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `PruneAndAssign(Vector<>)` | Prunes weights and assigns the important ones to the current task. |
| `Reset` |  |

