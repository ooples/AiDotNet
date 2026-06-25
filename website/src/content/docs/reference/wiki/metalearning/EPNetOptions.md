---
title: "EPNetOptions<T, TInput, TOutput>"
description: "Configuration options for EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020).

## For Beginners

EPNet refines features by spreading information:

1. Extract features for all examples (support + query)
2. Build a similarity graph connecting all examples
3. Propagate feature information through the graph
4. Features become more discriminative after propagation

It's like a game of telephone where each example improves by hearing from its neighbors.

## How It Works

EPNet propagates embeddings through a label propagation graph to refine features
using both support and query set information in a transductive manner.

Reference: Rodriguez, P., Laradji, I., Drouin, A., & Lacoste, A. (2020).
Embedding Propagation: Smoother Manifold for Few-Shot Classification. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EPNetOptions(IFullModel<,,>)` | Initializes a new instance of EPNetOptions. |

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
| `NumNeighbors` | Gets or sets the number of nearest neighbors for the propagation graph. |
| `OuterLearningRate` |  |
| `PropagationAlpha` | Gets or sets the propagation smoothing parameter (alpha). |
| `PropagationIterations` | Gets or sets the number of embedding propagation iterations. |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

