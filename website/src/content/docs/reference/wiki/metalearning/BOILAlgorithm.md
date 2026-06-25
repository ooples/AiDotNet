---
title: "BOILAlgorithm<T, TInput, TOutput>"
description: "Implementation of Body Only Inner Loop (BOIL) meta-learning algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Body Only Inner Loop (BOIL) meta-learning algorithm.

## For Beginners

Think of a neural network as having two jobs:

## How It Works

BOIL is the opposite of ANIL - it only adapts the feature extractor (body) during
inner-loop adaptation while keeping the classification head frozen. This explores
whether learning task-specific representations is more important than task-specific classifiers.

**Key Insight:** ANIL showed that adapting only the head works well, suggesting
the body learns general features. BOIL tests the complementary hypothesis: what if
we need to adapt HOW we see things (features) rather than HOW we decide (classifier)?

BOIL says: "The classifier is general enough - we just need to learn to SEE things
differently for each task!" So it only updates how the network extracts features,
while the decision-making layer stays fixed.

**Algorithm (MAML-style with body-only adaptation):**

Reference: Oh, J., Yoo, H., Kim, C., & Yun, S. Y. (2021).
BOIL: Towards Representation Change for Few-shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BOILAlgorithm(BOILOptions<,,>)` | Initializes a new instance of the BOILAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task by only updating the feature extractor (body). |
| `ApplyLayerwiseLearningRates(Vector<>)` | Applies layer-wise learning rate scaling to gradients. |
| `CloneBodyParameters` | Clones body parameters from the current model. |
| `CloneVector(Vector<>)` | Clones a vector. |
| `ComputeBodyGradients(,,Vector<>)` | Computes gradients for body parameters only. |
| `ComputeFirstOrderMetaGradients(IMetaLearningTask<,,>,Vector<>)` | Computes first-order meta-gradients. |
| `ComputeHeadGradients(,,Vector<>)` | Computes gradients for head parameters. |
| `ComputeL2Penalty(Vector<>)` | Computes L2 penalty for body parameters. |
| `ComputeLogits(Vector<>,Vector<>,Vector<>)` | Computes logits from features using head parameters. |
| `ComputeMetaGradients(IMetaLearningTask<,,>,Vector<>,)` | Computes meta-gradients for body and head. |
| `ComputeSecondOrderMetaGradients(IMetaLearningTask<,,>,Vector<>,)` | Computes second-order meta-gradients by differentiating through the inner adaptation loop. |
| `ConvertFromVector(Vector<>)` | Converts a vector to the output type. |
| `ForwardWithBody(,Vector<>)` | Performs forward pass with given body parameters and frozen head. |
| `InitializeBodyParameters` | Initializes body parameters. |
| `InitializeHeadBias` | Initializes head bias to zeros. |
| `InitializeHeadWeights` | Initializes head weights using Xavier initialization. |
| `InitializeParameters` | Initializes head and body parameter counts. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using BOIL's body-only adaptation approach. |
| `UpdateBodyInitialization(Vector<>)` | Updates body initialization using gradients. |

