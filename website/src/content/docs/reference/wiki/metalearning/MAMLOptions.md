---
title: "MAMLOptions<T, TInput, TOutput>"
description: "Configuration options for MAML (Model-Agnostic Meta-Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for MAML (Model-Agnostic Meta-Learning) algorithm.

## For Beginners

MAML has two learning loops:

- Inner loop: Fast adaptation to a specific task (uses InnerLearningRate, AdaptationSteps)
- Outer loop: Slow learning of good initialization (uses OuterLearningRate)

## How It Works

MAML learns an initialization that can be quickly fine-tuned to new tasks.
These options control both the inner loop (task adaptation) and outer loop (meta-optimization).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAMLOptions(IFullModel<,,>)` | Initializes a new instance of the MAMLOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of gradient steps to take during inner loop adaptation. |
| `CheckpointFrequency` | Gets or sets how often (in meta-iterations) to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints during meta-training. |
| `EvaluationFrequency` | Gets or sets how often (in meta-iterations) to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation during meta-training. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner-loop adaptation. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter updates (outer loop). |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations to perform. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-optimization). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation (FOMAML) instead of full MAML. |
| `UseFirstOrderApproximation` | Gets or sets whether to use first-order approximation (FOMAML). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the MAML options. |
| `IsValid` | Validates that all MAML configuration options are properly set. |

