---
title: "TADAMAlgorithm<T, TInput, TOutput>"
description: "Implementation of Task-Dependent Adaptive Metric (TADAM) algorithm for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Task-Dependent Adaptive Metric (TADAM) algorithm for few-shot learning.

## For Beginners

TADAM improves on ProtoNets by making the feature
extractor "aware" of the current task:

**How it works:**

1. Extract features from support set examples
2. Compute a "task embedding" summarizing what the task is about
3. Use this task embedding to adjust (condition) how features are extracted
4. Compute prototypes from the conditioned features
5. Classify queries using scaled distances to prototypes

**Key insight:** Different tasks may require focusing on different features.
TADAM learns to adjust what the network pays attention to based on the task.

## How It Works

TADAM extends Prototypical Networks by incorporating:

1. Task Conditioning (TC) using FiLM layers to modulate features
2. Metric Scaling to learn per-dimension distance weights
3. Auxiliary Co-Training for improved feature learning

**Algorithm - TADAM:**

**Key Innovations:**

1. **Task Conditioning (TC)**: FiLM layers modulate feature maps based on task context.

gamma and beta parameters are generated from the task embedding.

2. **Metric Scaling**: Learns per-dimension weights (alpha) for the distance metric,

allowing the model to emphasize or de-emphasize different feature dimensions.

3. **Learnable Temperature**: The temperature tau controls softmax sharpness and

is learned along with other parameters.

4. **Auxiliary Co-Training**: Optional auxiliary classification loss on base classes

to improve feature learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TADAMAlgorithm(TADAMOptions<,,>)` | Initializes a new instance of the TADAMAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `AddL2Regularization()` | Adds L2 regularization to the loss. |
| `ApplyFiLM(Tensor<>,Tensor<>,Tensor<>)` | Applies FiLM modulation: gamma * features + beta. |
| `ApplySoftmaxWithTemperature(Matrix<>)` | Applies softmax with learnable temperature. |
| `ComputeAuxiliaryLoss(IMetaLearningTask<,,>)` | Computes auxiliary co-training loss. |
| `ComputeConditionedPrototypes(,)` | Computes prototypes using task-conditioned features. |
| `ComputeCrossEntropyLoss(Matrix<>,)` | Computes cross-entropy loss. |
| `ComputeFeatureMean(Tensor<>)` | Computes mean feature vector. |
| `ComputeFiLMBeta(Tensor<>)` | Computes FiLM beta (shift) parameters from task embedding. |
| `ComputeFiLMGamma(Tensor<>)` | Computes FiLM gamma (scale) parameters from task embedding. |
| `ComputeMetricLoss(IMetaLearningTask<,,>)` | Computes metric-based loss for gradient computation. |
| `ComputeMetricScaleGradients(Tensor<>,Dictionary<Int32,Tensor<>>,)` | Computes analytical gradients for all metric scale parameters in a single pass. |
| `ComputePrototype(List<Tensor<>>)` | Computes prototype (mean) of a list of feature tensors. |
| `ComputeScaledDistances(Tensor<>,Dictionary<Int32,Tensor<>>)` | Computes scaled squared distances from queries to prototypes. |
| `ComputeTaskEmbedding(,)` | Computes task embedding from support set statistics. |
| `EncodeExamples()` | Encodes input examples to feature representations using the MetaModel. |
| `EncodeWithTaskConditioning(,Tensor<>)` | Encodes examples with FiLM-style task conditioning. |
| `GetClassLabel(,Int32)` | Gets class label from output at specified index. |
| `GroupAndComputePrototypes(Tensor<>,)` | Groups features by class and computes prototypes. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `NormalizeTensor(Tensor<>)` | Normalizes a tensor to unit length. |
| `TrainEpisode(IMetaLearningTask<,,>)` | Trains on a single episode using TADAM. |
| `UpdateMetricScale(IMetaLearningTask<,,>,)` | Updates metric scaling parameters using gradient descent with efficient batch computation. |
| `UpdateParameters(IMetaLearningTask<,,>,)` | Updates all learnable parameters using gradient descent. |
| `UpdateTemperature(IMetaLearningTask<,,>,)` | Updates temperature parameter using gradient descent. |

