---
title: "SEALOptions<T, TInput, TOutput>"
description: "Configuration options for the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm.

## For Beginners

SEAL learns the best starting point for a model so that
it can quickly adapt to new tasks with minimal data. Think of it like learning
how to learn - after seeing many tasks, the model knows how to pick up new skills
quickly.

Imagine learning to play musical instruments:

- Learning your first instrument (piano) is hard
- Learning your second instrument (guitar) is easier
- By your 5th instrument, you've learned principles that help you

pick up any new instrument much faster

SEAL does the same with machine learning models!

## How It Works

SEAL is a gradient-based meta-learning algorithm that combines ideas from MAML with
sample-efficiency improvements. It learns initial parameters that can be quickly
adapted to new tasks with just a few examples, incorporating temperature scaling,
entropy regularization, and optional adaptive learning rates.

Key features of SEAL:

- Temperature scaling: Controls confidence in predictions during meta-training
- Entropy regularization: Encourages diverse predictions to prevent overconfident models
- Adaptive learning rates: Per-parameter learning rate adaptation based on gradient norms
- Weight decay: Prevents overfitting to meta-training tasks

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SEALOptions(IFullModel<,,>)` | Initializes a new instance of the SEALOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of gradient steps to take during inner loop adaptation. |
| `AdaptiveLearningRateDecay` | Gets or sets the decay rate for running mean in adaptive learning rates. |
| `AdaptiveLearningRateEpsilon` | Gets or sets the epsilon value for numerical stability in adaptive learning rates. |
| `AdaptiveLearningRateMode` | Gets or sets the mode for adaptive learning rate computation. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EnableCheckpointing` | Gets or sets whether to save checkpoints during training. |
| `EntropyCoefficient` | Gets or sets the entropy regularization coefficient. |
| `EntropyOnlyDuringMetaTrain` | Gets or sets whether to apply entropy regularization only during meta-training. |
| `EvaluationFrequency` | Gets or sets how often to evaluate during meta-training. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (task adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner-loop adaptation. |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter updates (outer loop). |
| `MinTemperature` | Gets or sets the minimum temperature for temperature annealing. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations to perform. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-optimization). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `Temperature` | Gets or sets the temperature scaling factor for the loss function. |
| `UseAdaptiveInnerLR` | Gets or sets whether to use adaptive inner learning rates. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation (FOMAML-style). |
| `WeightDecay` | Gets or sets the weight decay (L2 regularization) coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the SEAL options. |
| `IsValid` | Validates that all SEAL configuration options are properly set. |

