---
title: "GradientCheckpointing<T>"
description: "Provides gradient checkpointing functionality for memory-efficient training."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Autodiff`

Provides gradient checkpointing functionality for memory-efficient training.

## For Beginners

When training large neural networks, storing all intermediate
results (activations) can use a lot of memory. Gradient checkpointing saves memory by:

1. Only storing activations at certain "checkpoints"
2. During backpropagation, recomputing the activations between checkpoints

This uses less memory but takes more time (roughly 30% more computation).
It's essential for training very large models that wouldn't otherwise fit in GPU memory.

## How It Works

Gradient checkpointing (also known as activation checkpointing or memory checkpointing)
is a technique that trades computation time for memory by not storing all intermediate
activations during the forward pass. Instead, it recomputes them during the backward pass.

This implementation follows patterns from PyTorch's torch.utils.checkpoint and
TensorFlow's tf.recompute_grad.

## Methods

| Method | Summary |
|:-----|:--------|
| `Checkpoint(Func<ComputationNode<>>,IEnumerable<ComputationNode<>>)` | Executes a function with gradient checkpointing. |
| `CheckpointMultiOutput(Func<IReadOnlyList<ComputationNode<>>>,IEnumerable<ComputationNode<>>)` | Executes a function with gradient checkpointing, supporting multiple outputs. |
| `EstimateMemorySavings(Int32,Int64,Int32)` | Estimates memory savings from using gradient checkpointing. |
| `SequentialCheckpoint(IReadOnlyList<Func<ComputationNode<>,ComputationNode<>>>,ComputationNode<>,Int32)` | Creates a sequential checkpoint that divides a sequence of layers into segments. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_checkpointStack` | Thread-local stack to track checkpoint boundaries during forward/backward passes. |

