---
title: "PTMAPOptions<T, TInput, TOutput>"
description: "Configuration options for PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021).

## For Beginners

PT+MAP is a surprisingly simple yet powerful approach:

1. Extract features using any pretrained backbone
2. Apply a power transform to normalize the feature distribution
3. Use MAP (optimal Bayesian) estimation to classify query examples transductively

The power transform makes features more Gaussian, which makes the simple
Bayesian classifier work much better. Simple math, strong results.

## How It Works

PT+MAP applies power transform normalization to features followed by MAP estimation
for transductive few-shot classification.

Reference: Hu, Y., Gripon, V., & Pateux, S. (2021).
Leveraging the Feature Distribution in Transfer-based Few-Shot Learning. ICLR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PTMAPOptions(IFullModel<,,>)` | Initializes a new instance of PTMAPOptions. |

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
| `LossFunction` | Gets or sets the loss function. |
| `MAPIterations` | Gets or sets the number of MAP estimation iterations. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `PowerTransformBeta` | Gets or sets the power transform exponent (beta). |
| `RandomSeed` | Gets or sets the random seed. |
| `Temperature` | Gets or sets the temperature for softmax in MAP estimation. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

