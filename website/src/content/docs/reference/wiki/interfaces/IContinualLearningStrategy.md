---
title: "IContinualLearningStrategy<T>"
description: "Defines a strategy for continual learning that helps neural networks learn multiple tasks sequentially without forgetting previously learned knowledge."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a strategy for continual learning that helps neural networks learn multiple tasks
sequentially without forgetting previously learned knowledge.

## For Beginners

Continual learning addresses a fundamental challenge in neural networks
called "catastrophic forgetting." When a neural network learns a new task, it often forgets
how to perform previous tasks. This happens because the network's weights are modified
to optimize for the new task, overwriting the knowledge from earlier tasks.

## How It Works

Continual learning strategies help networks learn multiple tasks sequentially while
preserving knowledge from previous tasks. Common approaches include:

**Typical Usage Flow:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` | Whether this strategy accumulates regularization strength across tasks. |
| `Lambda` | Gets the regularization strength parameter (lambda) for loss-based continual learning. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` | Processes information after completing training on a task. |
| `BeforeTask(INeuralNetwork<>,Int32)` | Prepares the strategy before starting to learn a new task. |
| `ComputeLoss(INeuralNetwork<>)` | Computes the regularization loss to prevent forgetting previous tasks. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` | Modifies the gradient to prevent catastrophic forgetting. |
| `Reset` | Resets the strategy, clearing all stored task information. |

