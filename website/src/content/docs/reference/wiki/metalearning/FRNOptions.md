---
title: "FRNOptions<T, TInput, TOutput>"
description: "Configuration options for FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021).

## For Beginners

FRN asks "can I rebuild this query from the support set?":

1. For each class, collect support feature maps
2. Try to reconstruct the query's feature map using ONLY features from class k
3. If reconstruction is good (low error): query likely belongs to class k
4. If reconstruction is bad (high error): query likely NOT class k

It's like asking: "Can I explain this new image using only the patterns I've seen
from cats?" If yes, it's probably a cat. If not, try the next class.

## How It Works

FRN classifies queries by reconstructing their feature maps from class-specific support
feature maps. The reconstruction error serves as the distance metric.

Reference: Wertheimer, D., Tang, L., & Hariharan, B. (2021).
Few-Shot Classification With Feature Map Reconstruction Networks. CVPR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FRNOptions(IFullModel<,,>)` | Initializes a new instance of FRNOptions. |

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
| `NumComponents` | Gets or sets the number of reconstruction components per class. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `ReconstructionLambda` | Gets or sets the reconstruction regularization parameter (lambda). |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

