---
title: "FewTUREOptions<T, TInput, TOutput>"
description: "Configuration options for FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation) (Hiller et al., ECCV 2022)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation) (Hiller et al., ECCV 2022).

## For Beginners

FewTURE combines transformers with uncertainty:

1. Uses a Vision Transformer (ViT) to get patch-level features (tokens)
2. Matches queries to support classes at the token/patch level
3. Estimates uncertainty for each prediction
4. Uncertain predictions are treated more carefully during classification

This is like comparing puzzle pieces: instead of comparing whole images,
compare individual patches and aggregate the evidence.

## How It Works

FewTURE uses vision transformers with token-level local features and uncertainty estimation
for few-shot classification. It operates on patch tokens from a ViT backbone.

Reference: Hiller, M., Ma, R., Harber, M., & Ommer, B. (2022).
Rethinking Generalization in Few-Shot Classification. ECCV 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FewTUREOptions(IFullModel<,,>)` | Initializes a new instance of FewTUREOptions. |

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
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `NumTokens` | Gets or sets the number of patch tokens to consider. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UncertaintyMethod` | Gets or sets the uncertainty estimation method. |
| `UncertaintyThreshold` | Gets or sets the uncertainty threshold for reliable estimation. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

