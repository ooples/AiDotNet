---
title: "ProtoNetsAlgorithm<T, TInput, TOutput>"
description: "Implementation of Prototypical Networks (ProtoNets) algorithm for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Prototypical Networks (ProtoNets) algorithm for few-shot learning.

## For Beginners

ProtoNets learns to recognize new classes from just a few examples:

**How it works:**

1. For each new class, create a "prototype" (average of all examples)
2. To classify a new example, find which prototype is closest
3. Distance is measured in a learned feature space
4. Uses soft nearest neighbor with learnable distance metric

**Simple example:**

- Support set: 3 images each of 5 different animal species (15 images total)
- Create prototype for each species by averaging their features
- Query image: classify by finding nearest animal prototype
- Learning: train encoder to make same-species images cluster together

## How It Works

Prototypical Networks learn a metric space where classification can be performed by computing
distances to prototype representations of each class. Each prototype is the mean vector of
the support set examples for that class.

**Algorithm - Prototypical Networks:**

**Key Insights:**

1. **Non-parametric Classification**: No classifier parameters to learn,

just need a good feature encoder. Prototypes are computed on-the-fly.

2. **Metric Learning**: The encoder learns to cluster same-class examples

and separate different classes in the feature space.

3. **Efficient Adaptation**: To adapt to new classes, just compute new

prototypes - no gradient updates needed!

4. **Interpretable**: Prototypes provide an intuitive representation of each

class as the "average example".

Reference: Snell, J., Swersky, K., & Zemel, R. (2017).
Prototypical Networks for Few-shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProtoNetsAlgorithm(ProtoNetsOptions<,,>)` | Initializes a new instance of the ProtoNetsAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task by computing class prototypes from the support set. |
| `ApplyAttentionWeights(Vector<>,Int32)` | Applies attention weights to enhance prototype computation. |
| `ApplyClassScaling(,Int32)` | Applies class-specific scaling to distance computation. |
| `ApplySoftmaxToDistances(Matrix<>)` | Applies softmax to distances to convert them to class probabilities. |
| `ApplyTemperatureScaling(Matrix<>)` | Applies temperature scaling to distances. |
| `ComputeClassPrototypes(Matrix<>,)` | Computes class prototypes by averaging features of examples from the same class. |
| `ComputeCosineDistance(Vector<>,Vector<>)` | Computes cosine distance between two feature vectors. |
| `ComputeCrossEntropyLoss(Matrix<>,)` | Computes cross-entropy loss between predicted probabilities and true labels. |
| `ComputeDistances(Matrix<>,Dictionary<Int32,Vector<>>)` | Computes distances between query features and class prototypes. |
| `ComputeMahalanobisDistance(Vector<>,Vector<>)` | Computes Mahalanobis distance using learned covariance scaling. |
| `ComputeMeanVector(List<Vector<>>)` | Computes the mean of a list of vectors. |
| `ComputeMultiDimIndex(Int32,Int32[],Int32)` | Converts a flat index to multi-dimensional tensor indices. |
| `ConvertToMatrix()` | Converts the output to a matrix format. |
| `EncodeExamples()` | Encodes input examples to feature space using the feature encoder. |
| `GetClassLabel(,Int32)` | Extracts class label from output at specified index. |
| `GetOptions` |  |
| `GetRow(Matrix<>,Int32)` | Gets a row from a matrix as a vector. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using ProtoNets' episodic training. |
| `NormalizeFeatures(Matrix<>)` | Applies feature normalization (L2 normalization). |
| `TensorToMatrix(Tensor<>)` | Converts a tensor to a matrix. |
| `TrainEpisode(IMetaLearningTask<,,>)` | Trains the feature encoder on a single episode. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_attentionWeights` | Attention weights for prototype enhancement (if enabled). |
| `_classScalingFactors` | Class-specific scaling factors for adaptive distance computation. |

