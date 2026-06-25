---
title: "ILearningRateScheduler"
description: "Interface for learning rate schedulers that adjust the learning rate during training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.LearningRateSchedulers`

Interface for learning rate schedulers that adjust the learning rate during training.

## For Beginners

The learning rate controls how big each step is when the model is learning.
A scheduler automatically adjusts this step size during training - typically starting with larger steps
to make fast progress, then smaller steps to fine-tune the solution. Think of it like driving:
you go faster on the highway (early training) and slow down as you approach your destination (later training).

## How It Works

Learning rate schedulers are essential for training neural networks effectively. They adjust
the learning rate according to various strategies, enabling better convergence and final performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseLearningRate` | Gets the base (initial) learning rate. |
| `CurrentLearningRate` | Gets the current learning rate. |
| `CurrentStep` | Gets the current step (iteration or epoch count depending on scheduler type). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLearningRateAtStep(Int32)` | Gets the learning rate for a specific step without advancing the scheduler. |
| `GetState` | Gets the scheduler state for serialization/checkpointing. |
| `LoadState(Dictionary<String,Object>)` | Loads the scheduler state from a checkpoint. |
| `Reset` | Resets the scheduler to its initial state. |
| `Step` | Advances the scheduler by one step and returns the new learning rate. |

