---
title: "ReptileOptions<T, TInput, TOutput>"
description: "Configuration options for the Reptile meta-learning algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the Reptile meta-learning algorithm.

## For Beginners

Reptile is the simplest meta-learning algorithm to understand:

1. Train on a task for several steps
2. Move the starting point slightly toward where you ended up
3. Repeat with many tasks

After seeing many tasks, your starting point becomes great for learning any new task!

## How It Works

Reptile is a simple first-order meta-learning algorithm that doesn't require computing
gradients through the adaptation process. Instead, it interpolates between current
meta-parameters and adapted parameters.

Key difference from MAML: Reptile doesn't compute gradients through adaptation.
This makes it much simpler to implement and faster to run, with competitive performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReptileOptions(IFullModel<,,>)` | Initializes a new instance of the ReptileOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of gradient steps to take during inner loop adaptation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints during training. |
| `EvaluationFrequency` | Gets or sets how often to evaluate during meta-training. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerBatches` | Gets or sets the number of inner batches per adaptation step. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner-loop adaptation. |
| `Interpolation` | Gets or sets the interpolation factor for meta-updates. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter updates (outer loop). |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations to perform. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-update). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseSerialUpdate` | Gets or sets whether to use the serial (single-task) or batched update variant. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the Reptile options. |
| `IsValid` | Validates that all Reptile configuration options are properly set. |

