---
title: "MCLOptions<T, TInput, TOutput>"
description: "Configuration options for MCL (Meta-learning with Contrastive Learning) few-shot method."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for MCL (Meta-learning with Contrastive Learning) few-shot method.

## For Beginners

MCL improves features with contrastive learning:

1. Standard meta-learning loss: Be good at few-shot tasks
2. Contrastive loss: Same-class examples should be close, different-class far apart
3. By combining both, the learned features are better organized in embedding space

Think of it as: meta-learning teaches HOW to use features for few-shot tasks,
while contrastive learning teaches features to BE more useful for comparison.

## How It Works

MCL combines episodic meta-learning with supervised contrastive learning to produce
features that are both discriminative and well-clustered in embedding space.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MCLOptions(IFullModel<,,>)` | Initializes a new instance of MCLOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `ContrastiveTemperature` | Gets or sets the temperature for contrastive loss. |
| `ContrastiveWeight` | Gets or sets the contrastive loss weight. |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `NumWays` | Gets or sets the number of ways (classes) in the few-shot task. |
| `OuterLearningRate` |  |
| `ProjectionDim` | Gets or sets the projection dimension for contrastive head. |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

