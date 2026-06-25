---
title: "MatchingNetworksAlgorithm<T, TInput, TOutput>"
description: "Implementation of Matching Networks for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Matching Networks for few-shot learning.

## For Beginners

Matching Networks learn to pay attention to similar examples:

**How it works:**

1. Encode all examples (support and query) with a shared encoder
2. For each query, compute attention weights with all support examples
3. Use cosine similarity or learned attention for weights
4. Predict weighted sum of support labels (soft nearest neighbor)

**Key insight:** The network learns how to compare examples during encoding,
making the similarity measure task-aware.

## How It Works

Matching Networks use attention mechanisms over the support set to classify
query examples. It computes a weighted sum of support labels where weights are
determined by an attention function that measures similarity between examples.

**Algorithm - Matching Networks:**

**Key Insights:**

1. **Task-Aware Embeddings**: The encoder learns to produce embeddings that

are meaningful for the specific classification task at hand.

2. **Differentiable Attention**: The attention mechanism is fully differentiable,

allowing end-to-end training of the encoder.

3. **No Adaptation Needed**: At test time, simply encode new examples and

apply the same attention mechanism - no gradient updates required.

Reference: Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. (2016).
Matching Networks for One Shot Learning. NeurIPS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatchingNetworksAlgorithm(MatchingNetworksOptions<,,>)` | Initializes a new instance of the MatchingNetworksAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task by caching support embeddings. |
| `ApplySoftmax(Vector<>)` | Computes negative Euclidean distance (higher = more similar). |
| `ComputeAttentionGradients(IMetaLearningTask<,,>,)` | Computes gradients with respect to the attention-based loss using finite differences. |
| `ComputeAttentionWeights(Vector<>,Matrix<>)` | Computes attention weights between query and all support examples. |
| `ComputeCrossEntropyLoss(Matrix<>,)` | Computes cross-entropy loss. |
| `ComputePredictions(Matrix<>,Matrix<>,)` | Computes predictions using attention mechanism. |
| `ConvertLabelsToOneHot(,Int32)` | Converts labels to one-hot encoded matrix. |
| `ConvertToMatrix()` | Converts output to matrix format. |
| `EncodeExamples()` | Encodes input examples to feature space. |
| `GetClassLabel(,Int32)` | Gets class label from output at specified index. |
| `GetOptions` |  |
| `GetRow(Matrix<>,Int32)` | Gets a row from a matrix as a vector. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using Matching Networks' episodic training. |
| `TensorToMatrix(Tensor<>)` | Converts tensor to matrix. |
| `TrainEpisode(IMetaLearningTask<,,>)` | Trains the encoder on a single episode. |

