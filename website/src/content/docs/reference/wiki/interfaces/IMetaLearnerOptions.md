---
title: "IMetaLearnerOptions<T>"
description: "Configuration options interface for meta-learning algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Configuration options interface for meta-learning algorithms.

## For Beginners

Think of meta-learning like learning to study effectively:

- **Inner loop:** How you study for a specific exam (practice problems, examples)
- **Outer loop:** Learning better study techniques by reflecting across many exams

The configuration controls both loops:

- InnerLearningRate: How aggressively to adapt to each task
- OuterLearningRate: How much to update the meta-parameters
- AdaptationSteps: How many gradient updates per task

## How It Works

Meta-learning algorithms use a two-loop optimization structure:

- **Inner loop:** Fast adaptation to a specific task using support set
- **Outer loop:** Meta-optimization to improve adaptation across all tasks

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets the number of gradient descent steps for inner loop adaptation. |
| `CheckpointFrequency` | Gets the checkpoint save frequency in meta-iterations. |
| `EnableCheckpointing` | Gets whether to save checkpoints during training. |
| `EvaluationFrequency` | Gets the evaluation frequency in meta-iterations. |
| `EvaluationTasks` | Gets the number of evaluation tasks for periodic validation. |
| `GradientClipThreshold` | Gets the gradient clipping threshold to prevent exploding gradients. |
| `InnerLearningRate` | Gets the inner loop learning rate for task-specific adaptation. |
| `MetaBatchSize` | Gets the number of tasks to sample per meta-update (meta-batch size). |
| `NumMetaIterations` | Gets the number of meta-training iterations to perform. |
| `OuterLearningRate` | Gets the outer loop learning rate for meta-optimization. |
| `RandomSeed` | Gets the random seed for reproducible task sampling and initialization. |
| `UseFirstOrder` | Gets whether to use first-order approximation (e.g., FOMAML, Reptile). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of this options instance. |
| `IsValid` | Validates that the configuration is valid and sensible. |

