---
title: "ANILAlgorithm<T, TInput, TOutput>"
description: "Implementation of Almost No Inner Loop (ANIL) meta-learning algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Almost No Inner Loop (ANIL) meta-learning algorithm.

## For Beginners

Think of a neural network as having two parts:

## How It Works

ANIL is a simplified version of MAML that only adapts the classification head
during inner-loop adaptation while keeping the feature extractor (body) frozen.
This significantly reduces computation while often maintaining competitive performance.

**Key Insight:** Most of the "learning to learn" ability in MAML comes from
learning a good feature representation, not from adapting the entire network.
By only adapting the final classification layer, ANIL achieves:

1. **Body (Feature Extractor):** Like learning to see and understand images
2. **Head (Classifier):** Like learning which button to press for each category

ANIL says: "The 'seeing' part is general enough - we just need to learn
which button to press for each new task!" So it only updates the button-pressing
part (head) and keeps the seeing part (body) fixed during adaptation.

**Algorithm (MAML-style with head-only adaptation):**

Reference: Raghu, A., Raghu, M., Bengio, S., & Vinyals, O. (2020).
Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ANILAlgorithm(ANILOptions<,,>)` | Initializes a new instance of the ANILAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task by only updating the classification head. |
| `CloneVector(Vector<>)` | Clones a vector. |
| `ComputeBodyGradients(,,Vector<>,Vector<>)` | Computes gradients for body parameters using the adapted head. |
| `ComputeFirstOrderMetaGradients(IMetaLearningTask<,,>,Vector<>,Vector<>)` | Computes first-order meta-gradients. |
| `ComputeHeadGradients(,,Vector<>,Vector<>)` | Computes gradients for head parameters only. |
| `ComputeL2Penalty(Vector<>)` | Computes L2 penalty for head weights. |
| `ComputeLogits(Vector<>,Vector<>,Vector<>)` | Computes logits from features using the head parameters. |
| `ComputeMetaGradients(IMetaLearningTask<,,>,Vector<>,Vector<>,)` | Computes meta-gradients for body and head initialization. |
| `ConvertFromVector(Vector<>)` | Converts a vector to the output type. |
| `ExtractFeatures()` | Extracts features using the frozen body of the model. |
| `ForwardWithHead(,Vector<>,Vector<>)` | Performs forward pass with custom head parameters. |
| `InitializeHeadBias` | Initializes head bias to zeros. |
| `InitializeHeadParameters` | Initializes the classification head parameters. |
| `InitializeHeadWeights` | Initializes head weights using Xavier/He initialization. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using ANIL's head-only adaptation approach. |
| `UpdateBodyParameters(Vector<>)` | Updates body parameters using gradients. |

