---
title: "ANILOptions<T, TInput, TOutput>"
description: "Configuration options for Almost No Inner Loop (ANIL) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Almost No Inner Loop (ANIL) algorithm.

## For Beginners

ANIL splits a neural network into two parts:

1. **Body (Feature Extractor):** Learns general features shared across tasks (FROZEN during adaptation)
2. **Head (Classifier):** Task-specific layer that is adapted for each new task

Key insight: Most of the "learning to learn" happens in the feature extractor,
which doesn't need to be adapted per-task. Only the small classifier head needs
to change for each new task.

**Benefits:**

- Much faster than MAML (fewer parameters to adapt)
- Less memory usage (no need to store gradients for body)
- Often performs as well as full MAML

## How It Works

ANIL is a simplified version of MAML that only adapts the classification head
during inner-loop adaptation, while keeping the feature extractor frozen.
This significantly reduces computation while maintaining competitive performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ANILOptions(IFullModel<,,>)` | Initializes a new instance of the ANILOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps (gradient steps on support set). |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `FeatureDimension` | Gets or sets the dimension of the final feature representation (before head). |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `HeadL2Regularization` | Gets or sets the L2 regularization strength for the head. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (head adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates (head only). |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter (outer loop) updates. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-update). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `ReinitializeHead` | Gets or sets whether to reinitialize the head for each task. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseHeadBias` | Gets or sets whether to use a bias term in the classification head. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the ANIL options. |
| `IsValid` | Validates that all ANIL configuration options are properly set. |

