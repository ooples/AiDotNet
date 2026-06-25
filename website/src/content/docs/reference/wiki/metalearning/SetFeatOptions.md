---
title: "SetFeatOptions<T, TInput, TOutput>"
description: "Configuration options for SetFeat (set-feature based few-shot learning) (Afrasiyabi et al., CVPR 2022)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for SetFeat (set-feature based few-shot learning) (Afrasiyabi et al., CVPR 2022).

## For Beginners

SetFeat treats each class as a SET, not just a point:

- ProtoNets: Each class = average of its examples (one point)
- SetFeat: Each class = rich representation capturing variety across examples

This helps because knowing HOW a class varies (e.g., small vs. large dogs)
is as important as knowing the class average.

## How It Works

SetFeat learns set-level features by processing each class's support examples as a set
rather than individual instances. A set encoder computes class representations that
capture intra-class variation, not just the class mean.

Reference: Afrasiyabi, A., Larochelle, H., Lalonde, J.F., & Gagne, C. (2022).
Matching Feature Sets for Few-Shot Image Classification. CVPR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SetFeatOptions(IFullModel<,,>)` | Initializes a new instance of SetFeatOptions. |

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
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `SetEncoderDim` | Gets or sets the set encoder dimension. |
| `UseCrossAttention` | Gets or sets whether to use cross-attention between support sets. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

