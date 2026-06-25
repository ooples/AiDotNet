---
title: "NPBMLOptions<T, TInput, TOutput>"
description: "Configuration options for NPBML (Neural Process-Based Meta-Learning)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for NPBML (Neural Process-Based Meta-Learning).

## For Beginners

NPBML is a probabilistic meta-learner:

1. Encodes support examples into a latent distribution (not just a point)
2. Samples from this distribution to capture task uncertainty
3. Decodes predictions for query examples using the sampled task representation

This means NPBML can say "I'm not sure about this task" by producing high-variance
predictions when the support set is ambiguous.

## How It Works

NPBML combines neural processes with meta-learning for probabilistic few-shot prediction.
It models the predictive distribution conditioned on the support set using a stochastic
latent variable that captures task-level uncertainty.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NPBMLOptions(IFullModel<,,>)` | Initializes a new instance of NPBMLOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `KLWeight` | Gets or sets the KL divergence weight. |
| `LatentDim` | Gets or sets the latent dimension for the task representation. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `NumSamples` | Gets or sets the number of samples for prediction averaging. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

