---
title: "TADAMOptions<T, TInput, TOutput>"
description: "Configuration options for Task-Dependent Adaptive Metric (TADAM) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Task-Dependent Adaptive Metric (TADAM) algorithm.

## For Beginners

TADAM improves on Prototypical Networks by:

1. **Task Conditioning (TC):** Adjusts features based on the specific task
2. **Metric Scaling:** Learns how to weight different feature dimensions
3. **Auxiliary Co-Training:** Uses additional classification to improve features

Think of it as ProtoNets that "pay attention" to what matters for each specific task.

## How It Works

TADAM extends prototypical networks by incorporating task-dependent metric learning.
It uses task conditioning (TC) to modulate the feature extraction process based on
the task at hand, and metric scaling to adapt distances in embedding space.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TADAMOptions(IFullModel<,,>)` | Initializes a new instance of the TADAMOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps. |
| `AuxiliaryLossWeight` | Gets or sets the weight for auxiliary loss. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EmbeddingDimension` | Gets or sets the embedding dimension. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InitialTemperature` | Gets or sets the initial value for the learnable temperature parameter. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (not used in TADAM). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates. |
| `L2Regularization` | Gets or sets the L2 regularization strength. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (feature encoder) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for network updates. |
| `NormalizeEmbeddings` | Gets or sets whether to normalize embeddings. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (encoder training). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `TaskEmbeddingDimension` | Gets or sets the dimension of task embeddings. |
| `UseAuxiliaryCoTraining` | Gets or sets whether to use auxiliary co-training. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseMetricScaling` | Gets or sets whether to use metric scaling. |
| `UseTaskConditioning` | Gets or sets whether to use task conditioning (FiLM layers). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the TADAM options. |
| `IsValid` | Validates that all TADAM configuration options are properly set. |

