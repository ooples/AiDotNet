---
title: "ProtoNetsOptions<T, TInput, TOutput>"
description: "Configuration options for Prototypical Networks (ProtoNets) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Prototypical Networks (ProtoNets) algorithm.

## For Beginners

ProtoNets is one of the simplest and most effective
few-shot learning methods:

1. Use a neural network to convert images/data into feature vectors
2. For each class in a task, compute the "prototype" (average feature vector)
3. To classify a new example, find the nearest prototype
4. Train the network to make same-class examples cluster together

Unlike MAML, ProtoNets doesn't need gradient updates at test time - just compute
prototypes and measure distances!

## How It Works

Prototypical Networks learn a metric space where classification is performed by computing
distances to prototype representations of each class. Each prototype is the mean vector of
the support set examples for that class in the learned embedding space.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProtoNetsOptions(IFullModel<,,>)` | Initializes a new instance of the ProtoNetsOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `DistanceFunction` | Gets or sets the distance function for measuring similarity between embeddings. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `LossFunction` | Gets or sets the loss function for training. |
| `MahalanobisScaling` | Gets or sets the scaling factor for Mahalanobis distance. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (feature encoder) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for feature encoder updates. |
| `NormalizeFeatures` | Gets or sets whether to L2-normalize feature embeddings. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (encoder training). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `Temperature` | Gets or sets the temperature for softmax scaling. |
| `UseAdaptiveClassScaling` | Gets or sets whether to use adaptive class-specific scaling factors. |
| `UseAttentionMechanism` | Gets or sets whether to use an attention mechanism for prototype computation. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the ProtoNets options. |
| `IsValid` | Validates that all ProtoNets configuration options are properly set. |

