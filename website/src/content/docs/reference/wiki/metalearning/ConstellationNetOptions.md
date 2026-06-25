---
title: "ConstellationNetOptions<T, TInput, TOutput>"
description: "Configuration options for ConstellationNet (Xu et al., ICLR 2021)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for ConstellationNet (Xu et al., ICLR 2021).

## For Beginners

ConstellationNet finds patterns in how parts relate:

1. Detect discriminative parts/regions in each example
2. Learn how these parts are arranged (their "constellation")
3. Compare queries to support classes by matching constellation patterns

Imagine recognizing a face: you don't just look at eyes, nose, mouth separately.
You also look at HOW they're arranged - that's the constellation.

## How It Works

ConstellationNet learns structured representations where parts and their spatial
relationships form "constellations" for few-shot classification.

Reference: Xu, C., Fu, Y., Liu, C., Wang, C., Li, J., Huang, F., Zhang, L., & Xue, X. (2021).
Learning Dynamic Alignment via Meta-filter for Few-shot Learning. CVPR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConstellationNetOptions(IFullModel<,,>)` | Initializes a new instance of ConstellationNetOptions. |

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
| `NumParts` | Gets or sets the number of constellation parts to detect. |
| `OuterLearningRate` |  |
| `PartFeatureDim` | Gets or sets the part feature dimension. |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |
| `UseSpatialRelations` | Gets or sets whether to use spatial relationships between parts. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

