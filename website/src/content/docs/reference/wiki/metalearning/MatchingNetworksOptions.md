---
title: "MatchingNetworksOptions<T, TInput, TOutput>"
description: "Configuration options for Matching Networks algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Matching Networks algorithm.

## For Beginners

Matching Networks learn to pay attention to similar examples:

1. Encode all examples (support and query) with a shared encoder
2. For each query, compute attention weights with all support examples
3. Use similarity (cosine, dot product, etc.) for weights
4. Predict weighted sum of support labels (soft nearest neighbor)

Unlike ProtoNets which uses class prototypes, Matching Networks consider
every support example individually.

## How It Works

Matching Networks use attention mechanisms over the support set to classify
query examples. It computes a weighted sum of support labels where weights are
determined by an attention function that measures similarity between examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatchingNetworksOptions(IFullModel<,,>)` | Initializes a new instance of the MatchingNetworksOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps. |
| `AttentionFunction` | Gets or sets the attention function for computing similarity. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (not used in Matching Networks). |
| `L2Regularization` | Gets or sets the L2 regularization strength. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (encoder) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for encoder updates. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (encoder training). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `Temperature` | Gets or sets the temperature for softmax attention. |
| `UseBidirectionalEncoding` | Gets or sets whether to use bidirectional encoding. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseFullContextEmbedding` | Gets or sets whether to use full context embedding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the Matching Networks options. |
| `IsValid` | Validates that all Matching Networks configuration options are properly set. |

