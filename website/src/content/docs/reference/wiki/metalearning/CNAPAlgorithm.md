---
title: "CNAPAlgorithm<T, TInput, TOutput>"
description: "Implementation of Conditional Neural Adaptive Processes (CNAP) for meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Conditional Neural Adaptive Processes (CNAP) for meta-learning.

## For Beginners

CNAP is like having a teacher who can instantly understand
a new subject from a few examples:

## How It Works

CNAP extends Neural Processes by conditioning on task-specific context points
and learning to produce fast adaptation weights for each task. Unlike MAML,
which adapts through gradient descent, CNAP learns to directly generate
task-specific weight modifications from context examples.

**Key Innovation:** Instead of computing gradients on support sets, CNAP:

1. Encodes support set examples into a task representation
2. Uses an adaptation network to generate task-specific fast weights
3. Applies these fast weights to modify the base model for that task

- MAML: Learns how to learn quickly (by finding a good starting point)
- CNAP: Learns to directly generate solutions (by understanding the task)

Imagine showing someone a few examples of a new font. CNAP doesn't need to
practice writing in that font (like MAML would). Instead, it understands
the pattern from examples and directly modifies how it writes.

**Architecture:**

- **Encoder:** Processes each context (input, output) pair into embeddings
- **Aggregator:** Combines embeddings into a single task representation
- **Adaptation Network:** Generates fast weights from task representation
- **Base Model:** Modified by fast weights to perform the task

**Algorithm:**

Reference: Requeima, J., Gordon, J., Bronskill, J., Nowozin, S., & Turner, R. E. (2019).
Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CNAPAlgorithm(CNAPOptions<,,>)` | Initializes a new instance of the CNAPAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using CNAP's feed-forward approach. |
| `ApplyEncoderTransform(Vector<>)` | Applies learned encoder transformation to the representation. |
| `ApplyFastWeights(IFullModel<,,>,Vector<>)` | Applies fast weights to modify the model parameters. |
| `ComputeBaseModelGradients(TaskBatch<,,>)` | Computes gradients for the base model parameters. |
| `ComputeMeanPartial(Vector<>,Int32)` | Computes mean of first n elements of a vector. |
| `ComputeNetworkGradients(IMetaLearningTask<,,>,Vector<>,Vector<>,IFullModel<,,>,)` | Computes gradients for the encoder and adaptation networks using finite differences. |
| `EncodeTask(,)` | Encodes support set data into a task representation vector. |
| `EstimateInputOutputDimension` | Estimates the input/output dimension for the encoder. |
| `GenerateFastWeights(Vector<>)` | Generates fast weights from the task representation. |
| `InitializeWeights(Int32)` | Initializes weights using Xavier/He initialization. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using CNAP's feed-forward adaptation approach. |
| `NormalizeFastWeights(Vector<>)` | Normalizes fast weights to prevent explosion. |
| `ResizeFastWeights(Vector<>,Int32)` | Resizes fast weights to match model parameter count. |

