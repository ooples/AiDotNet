---
title: "SEALAlgorithm<T, TInput, TOutput>"
description: "Implementation of the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm.

## For Beginners

SEAL learns the best starting point for a model so that
it can quickly adapt to new tasks with minimal data.

Imagine learning to play musical instruments:

- Learning your first instrument (e.g., piano) is hard
- Learning your second instrument (e.g., guitar) is easier
- By the time you learn your 5th instrument, you've learned principles of music

that help you pick up new instruments much faster

SEAL does the same with machine learning models - it learns from many tasks
to find a great starting point that makes adapting to new tasks much faster.

## How It Works

SEAL is a gradient-based meta-learning algorithm that combines ideas from MAML with
sample-efficiency improvements. It learns initial parameters that can be quickly
adapted to new tasks with just a few examples.

**Key Features:**

- Temperature scaling: Controls confidence in predictions during meta-training
- Entropy regularization: Encourages diverse predictions to prevent overconfident models
- Adaptive learning rates: Per-parameter learning rate adaptation based on gradient norms
- Weight decay: Prevents overfitting to meta-training tasks

**Algorithm:**

1. Sample a batch of tasks
2. For each task:

a. Clone the meta-model
b. Adapt to the task using support set (inner loop)
c. Evaluate on query set to compute meta-loss
d. Apply temperature scaling and entropy regularization
e. Compute meta-gradients

3. Average meta-gradients across tasks
4. Apply weight decay and update meta-parameters

Reference: Based on gradient-based meta-learning with additional efficiency
improvements including temperature scaling and entropy regularization.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SEALAlgorithm(SEALOptions<,,>)` | Initializes a new instance of the SEALAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using gradient descent. |
| `ApplyAdaptiveGradients(Vector<>,Vector<>,Vector<>)` | Applies gradients with per-parameter adaptive learning rates. |
| `ComputeAdaptiveLearningRates(Vector<>,Vector<>)` | Computes adaptive learning rates based on gradient statistics. |
| `ComputeCurrentTemperature` | Computes the current temperature based on annealing schedule. |
| `ComputeEntropyRegularization()` | Computes entropy regularization term for the predictions. |
| `ComputeMetaGradients(IFullModel<,,>,IMetaLearningTask<,,>,Vector<>)` | Computes meta-gradients for the outer loop update. |
| `ConvertToProbabilities()` | Converts model outputs to probability distribution using softmax. |
| `InnerLoopAdaptation(IFullModel<,,>,IMetaLearningTask<,,>)` | Performs the inner loop adaptation to a specific task. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using SEAL's sample-efficient approach. |

