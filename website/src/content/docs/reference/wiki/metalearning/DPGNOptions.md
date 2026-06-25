---
title: "DPGNOptions<T, TInput, TOutput>"
description: "Configuration options for DPGN (Distribution Propagation Graph Network) (Yang et al., CVPR 2020)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for DPGN (Distribution Propagation Graph Network) (Yang et al., CVPR 2020).

## For Beginners

DPGN builds two graphs to pass information:

1. Point graph: Passes feature information between examples (like GNN meta-learning)
2. Distribution graph: Passes uncertainty/confidence information
3. Both graphs refine each other iteratively

The dual approach captures both "what features are similar" and
"how confident we are about similarities."

## How It Works

DPGN uses a dual graph structure to propagate both point estimates and distribution
information between support and query examples for few-shot classification.

Reference: Yang, L., Li, L., Zhang, Z., Zhou, X., Zhou, E., & Liu, Y. (2020).
DPGN: Distribution Propagation Graph Network for Few-Shot Learning. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPGNOptions(IFullModel<,,>)` | Initializes a new instance of DPGNOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EdgeFeatureDim` | Gets or sets the edge feature dimension. |
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
| `NodeFeatureDim` | Gets or sets the graph node feature dimension. |
| `NumMetaIterations` |  |
| `NumPropagationLayers` | Gets or sets the number of graph propagation layers. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

