---
title: "CNAPOptions<T, TInput, TOutput>"
description: "Configuration options for the Conditional Neural Adaptive Processes (CNAP) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the Conditional Neural Adaptive Processes (CNAP) algorithm.

## For Beginners

CNAP learns to understand tasks from examples:

1. It encodes examples (context points) into a task representation
2. It uses this representation to generate task-specific fast weights
3. The fast weights modify the base model to work well on that specific task

This is like having a teacher who can quickly understand what kind of problem
you're working on and adjust their teaching style accordingly.

## How It Works

CNAP extends Neural Processes by conditioning on task-specific context points
and learning to produce fast adaptation weights for each task. This enables
effective few-shot learning through learned task representations.

**Key Components:**

- **Encoder:** Processes context points into task representations
- **Adaptation Network:** Generates task-specific fast weights from representations
- **Base Model:** The neural network that gets modified by fast weights

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CNAPOptions(IFullModel<,,>)` | Initializes a new instance of the CNAPOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationNetworkLayers` | Gets or sets the number of layers in the adaptation network. |
| `AdaptationSteps` | Gets or sets the number of gradient steps to take during inner loop adaptation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `ContextSize` | Gets or sets the number of context points to use for encoding (support set size). |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints during training. |
| `EncoderLayers` | Gets or sets the number of layers in the encoder network. |
| `EvaluationFrequency` | Gets or sets how often to evaluate during meta-training. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `FastWeightMode` | Gets or sets how fast weights are applied to the model. |
| `FastWeightRegularization` | Gets or sets the regularization weight for fast weights. |
| `FastWeightScale` | Gets or sets the scale factor for fast weights. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `HiddenDimension` | Gets or sets the hidden dimension for the encoder and adaptation networks. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner-loop adaptation. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter updates (outer loop). |
| `NormalizeFastWeights` | Gets or sets whether to normalize fast weights. |
| `NumAttentionHeads` | Gets or sets the number of attention heads when using attention. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations to perform. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-update). |
| `PredictUncertainty` | Gets or sets whether to predict uncertainty along with predictions. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `RepresentationDimension` | Gets or sets the dimension of the task representation vector. |
| `UncertaintyWeight` | Gets or sets the weight for uncertainty loss when predicting uncertainty. |
| `UseAttention` | Gets or sets whether to use attention for aggregating context points. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseLayerNorm` | Gets or sets whether to use layer normalization in networks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the CNAP options. |
| `IsValid` | Validates that all CNAP configuration options are properly set. |

