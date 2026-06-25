---
title: "LEOOptions<T, TInput, TOutput>"
description: "Configuration options for Latent Embedding Optimization (LEO) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Latent Embedding Optimization (LEO) algorithm.

## For Beginners

Imagine you have a very large model with millions of parameters.
Updating all of them during few-shot learning is slow and can lead to overfitting.
LEO learns to "compress" these parameters into a much smaller space (like 64 numbers instead
of millions). Adaptation happens in this compressed space, which is faster and more
robust to overfitting.

## How It Works

LEO performs meta-learning by learning a low-dimensional latent space for model parameters.
Instead of adapting the full model parameters directly (like MAML), LEO:

**Key Insight:** Not all parameter configurations make sense for neural networks.
By learning a latent space, LEO restricts adaptation to the "manifold" of sensible
parameter settings, preventing bad updates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LEOOptions(IFullModel<,,>)` | Initializes a new instance of the LEOOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps in latent space. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `DropoutRate` | Gets or sets the dropout rate for the encoder and decoder. |
| `EmbeddingDimension` | Gets or sets the dimension of the feature embedding. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EntropyWeight` | Gets or sets the entropy weight for regularizing the decoder. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `HiddenDimension` | Gets or sets the hidden dimension for encoder/decoder networks. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (latent space adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates. |
| `KLWeight` | Gets or sets the KL divergence weight for the latent space regularization. |
| `L2Regularization` | Gets or sets the L2 regularization strength. |
| `LatentDimension` | Gets or sets the dimensionality of the latent space. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (feature encoder) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter (outer loop) updates. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-update). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `ShareEncoder` | Gets or sets whether to share the encoder across all classes. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseOrthogonalInit` | Gets or sets whether to use orthogonal initialization for the decoder. |
| `UseRelationEncoder` | Gets or sets whether to use a relation network for encoding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the LEO options. |
| `IsValid` | Validates that all LEO configuration options are properly set. |

