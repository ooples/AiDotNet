---
title: "RelationNetworkOptions<T, TInput, TOutput>"
description: "Configuration options for Relation Networks algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Relation Networks algorithm.

## For Beginners

Relation Networks learns how to compare examples:

1. Encode all examples (support and query) with a feature encoder
2. For each query, concatenate with each support example's features
3. Pass concatenated features through a relation module (neural network)
4. The relation module outputs a similarity score
5. Apply softmax to get class probabilities

Instead of using predefined distances (like Euclidean), it learns a neural
network to measure "how related" two examples are.

## How It Works

Relation Networks learn to compare query examples with class examples by learning
a relation function that measures similarity. Unlike metric learning approaches
that use fixed distance functions, Relation Networks learn the relation function
end-to-end.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RelationNetworkOptions(IFullModel<,,>)` | Initializes a new instance of the RelationNetworkOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps. |
| `AggregationMethod` | Gets or sets the aggregation method for combining multiple support example scores. |
| `ApplyFeatureTransform` | Gets or sets whether to apply feature transformation before relation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `ConcatenationDimension` | Gets or sets the dimension for feature concatenation. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `FeatureEncoderL2Reg` | Gets or sets the L2 regularization strength for the feature encoder. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (not used in Relation Networks). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model (feature encoder) to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for network updates. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumHeads` | Gets or sets the number of heads for multi-head relation. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (encoder and relation module training). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `RelationDropout` | Gets or sets the dropout rate for the relation module. |
| `RelationHiddenDimension` | Gets or sets the hidden dimension for the relation module. |
| `RelationModuleL2Reg` | Gets or sets the L2 regularization strength for the relation module. |
| `RelationType` | Gets or sets the type of relation module architecture. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseMultiHeadRelation` | Gets or sets whether to use multi-head relation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the Relation Network options. |
| `IsValid` | Validates that all Relation Network configuration options are properly set. |

