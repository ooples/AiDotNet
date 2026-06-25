---
title: "FEATOptions<T, TInput, TOutput>"
description: "Configuration options for FEAT (Few-shot Embedding Adaptation with Transformer) (Ye et al., CVPR 2020)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for FEAT (Few-shot Embedding Adaptation with Transformer) (Ye et al., CVPR 2020).

## For Beginners

FEAT improves upon ProtoNets by making prototypes "task-aware":

**Problem with ProtoNets:**
Each class prototype is computed independently (mean of support features).
But knowing about OTHER classes in the task could help: if class A and B are similar,
their prototypes should be pushed apart.

**FEAT's solution:**
Use a transformer to let prototypes "talk to each other":

1. Compute initial prototypes (like ProtoNets)
2. Feed all prototypes through a set-to-set transformer
3. The transformer adjusts each prototype based on the others
4. Result: task-adapted prototypes that are more discriminative

**Analogy:**
Imagine placing labels on a map:

- ProtoNets: Places each label independently
- FEAT: Adjusts labels so they're evenly spread and don't overlap

## How It Works

FEAT adapts embeddings to be task-specific using a set-to-set transformer that takes
class prototypes as input and outputs task-adapted prototypes. The transformer captures
inter-class relationships to produce more discriminative prototypes.

Reference: Ye, H.J., Hu, H., Zhan, D.C., & Sha, F. (2020).
Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FEATOptions(IFullModel<,,>)` | Initializes a new instance of FEATOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `ContrastiveWeight` | Gets or sets the balance weight between contrastive loss and classification loss. |
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
| `NumTransformerHeads` | Gets or sets the number of attention heads in the set-to-set transformer. |
| `NumTransformerLayers` | Gets or sets the number of transformer layers. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `Temperature` | Gets or sets the temperature for similarity computation. |
| `TransformerDim` | Gets or sets the feature dimension for the set-to-set transformer. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

