---
title: "BOILOptions<T, TInput, TOutput>"
description: "Configuration options for Body Only Inner Loop (BOIL) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Body Only Inner Loop (BOIL) algorithm.

## For Beginners

BOIL splits a neural network into two parts:

## How It Works

BOIL is the opposite of ANIL - it only adapts the feature extractor (body) during
inner-loop adaptation while keeping the classification head frozen. This explores
the hypothesis that task-specific features are more important than task-specific classifiers.

This is the opposite of ANIL (which freezes body, adapts head). BOIL tests whether
it's better to adapt HOW we see things rather than HOW we decide.

**When to use BOIL:**

- When tasks differ more in their visual/input patterns than their decision boundaries
- When you have a good meta-learned classifier that works across tasks
- When you want to experiment with different adaptation strategies

Reference: Oh, J., Yoo, H., Kim, C., & Yun, S. Y. (2021).
BOIL: Towards Representation Change for Few-shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BOILOptions(IFullModel<,,>)` | Initializes a new instance of the BOILOptions class with the required meta-model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` | Gets or sets the number of adaptation steps (gradient steps on support set). |
| `BodyAdaptationFraction` | Gets or sets the fraction of body parameters to adapt (for efficiency). |
| `BodyL2Regularization` | Gets or sets the L2 regularization strength for the body. |
| `CheckpointFrequency` | Gets or sets how often to save checkpoints. |
| `DataLoader` | Gets or sets the episodic data loader for sampling tasks. |
| `EarlyLayerLrMultiplier` | Gets or sets the learning rate multiplier for earlier layers (if using layerwise rates). |
| `EnableCheckpointing` | Gets or sets whether to save model checkpoints. |
| `EvaluationFrequency` | Gets or sets how often to evaluate the meta-learner. |
| `EvaluationTasks` | Gets or sets the number of tasks to use for evaluation. |
| `FeatureDimension` | Gets or sets the dimension of the final feature representation (before head). |
| `GradientClipThreshold` | Gets or sets the maximum gradient norm for gradient clipping. |
| `InnerLearningRate` | Gets or sets the learning rate for the inner loop (body adaptation). |
| `InnerOptimizer` | Gets or sets the optimizer for inner loop updates (body only). |
| `LossFunction` | Gets or sets the loss function for training. |
| `MetaBatchSize` | Gets or sets the number of tasks to sample per meta-training iteration. |
| `MetaModel` | Gets or sets the meta-model to be trained. |
| `MetaOptimizer` | Gets or sets the optimizer for meta-parameter (outer loop) updates. |
| `NumClasses` | Gets or sets the number of output classes. |
| `NumMetaIterations` | Gets or sets the total number of meta-training iterations. |
| `OuterLearningRate` | Gets or sets the learning rate for the outer loop (meta-update). |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `ReinitializeBody` | Gets or sets whether to reinitialize the body for each task. |
| `UseFirstOrder` | Gets or sets whether to use first-order approximation. |
| `UseLayerwiseLearningRates` | Gets or sets whether to use layer-wise learning rates for the body. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the BOIL options. |
| `IsValid` | Validates that all BOIL configuration options are properly set. |

