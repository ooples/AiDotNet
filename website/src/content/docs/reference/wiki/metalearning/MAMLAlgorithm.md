---
title: "MAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of the MAML (Model-Agnostic Meta-Learning) algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of the MAML (Model-Agnostic Meta-Learning) algorithm.

## For Beginners

MAML is like teaching someone how to learn quickly.

Normal machine learning: Train a model for one specific task
MAML: Train a model to be easily trainable for many different tasks

It's like learning how to learn - by practicing on many tasks, the model
learns what kind of parameters make it easy to adapt to new tasks quickly.

## How It Works

MAML (Model-Agnostic Meta-Learning) is a meta-learning algorithm that trains models
to be easily fine-tunable. It learns initial parameters such that a small number of
gradient steps on a new task will lead to good performance.

Key features:

- Model-agnostic: works with any model trainable with gradient descent
- Learns good initialization rather than learning a fixed feature extractor
- Enables few-shot learning with just 1-5 examples per class

Reference: Finn, C., Abbeel, P., & Levine, S. (2017).
Model-agnostic meta-learning for fast adaptation of deep networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAMLAlgorithm(MAMLOptions<,,>)` | Initializes a new instance of the MAMLAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using MAML's inner loop optimization. |
| `BuildAdaptationStepsList(IMetaLearningTask<,,>)` | Builds the list of adaptation steps for second-order gradient computation. |
| `ComputeMetaGradients(Vector<>,IMetaLearningTask<,,>)` | Computes meta-gradients for the outer loop update. |
| `InnerLoopAdaptation(IFullModel<,,>,IMetaLearningTask<,,>)` | Performs the inner loop adaptation to a specific task. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using MAML's bi-level optimization. |

