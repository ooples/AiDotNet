---
title: "RelationNetworkAlgorithm<T, TInput, TOutput>"
description: "Implementation of Relation Networks algorithm for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Relation Networks algorithm for few-shot learning.

## For Beginners

Relation Networks learns how to compare examples:

**How it works:**

1. Encode all examples (support and query) with a feature encoder
2. For each query, concatenate with each support example's features
3. Pass concatenated features through a relation module (neural network)
4. The relation module outputs a similarity score
5. Apply softmax to get class probabilities

**Key insight:** Instead of using predefined distances (like Euclidean),
it learns a neural network to measure "how related" two examples are.

## How It Works

Relation Networks learn to compare query examples with class examples by learning
a relation function that measures similarity. Unlike metric learning approaches
that use fixed distance functions, Relation Networks learn the relation function
end-to-end.

**Algorithm - Relation Networks:**

**Key Insights:**

1. **Learnable Relation Function**: Instead of fixed distances, learns a neural

network to measure similarity. Can capture complex, non-linear relations.

2. **End-to-End Training**: Both feature encoder and relation module are

trained jointly, optimizing for the final classification task.

3. **Flexible Relations**: The relation module can learn to attend to

specific features, ignore noise, and detect subtle patterns.

4. **Scalable Complexity**: More powerful relation modules can handle

more complex tasks at the cost of computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RelationNetworkAlgorithm(RelationNetworkOptions<,,>)` | Initializes a new instance of the RelationNetworkAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `AddRegularizationTerms()` | Adds regularization terms to the loss. |
| `ApplySoftmaxToScores(Matrix<>)` | Applies softmax to relation scores to get class probabilities. |
| `ComputeClassRelationScore(Tensor<>,List<Tensor<>>)` | Computes relation score between a query and all examples in a class. |
| `ComputeCrossEntropyLoss(Matrix<>,)` | Computes cross-entropy loss between probabilities and true labels. |
| `ComputeL2Regularization(Vector<>)` | Computes L2 regularization for parameters. |
| `ComputeRelationScores(Tensor<>,Dictionary<Int32,List<Tensor<>>>)` | Computes relation scores between queries and class support examples. |
| `ComputeSingleRelationScore(Tensor<>,Tensor<>)` | Computes relation score between two feature tensors. |
| `ConcatenateFeatures(Tensor<>,Tensor<>)` | Concatenates two feature tensors. |
| `EncodeExamples()` | Encodes input examples to feature representations. |
| `GroupFeaturesByClass(Tensor<>,)` | Groups support features by their class labels. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `TrainEpisode(IMetaLearningTask<,,>)` | Trains the feature encoder and relation module on a single episode. |
| `UpdateNetworks(IMetaLearningTask<,,>)` | Updates both feature encoder and relation module parameters using gradient descent. |

