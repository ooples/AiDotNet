---
title: "CAMLOptions<T, TInput, TOutput>"
description: "Configuration options for CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023).

## For Beginners

CAML leverages powerful pretrained models:

1. Uses a large pretrained model (e.g., CLIP, DINO) to extract features
2. No fine-tuning of the backbone at all - features are frozen
3. A lightweight context module adapts features based on the support set
4. Classification is done by comparing adapted query features to prototypes

The key insight is that modern pretrained models produce such good features
that complex meta-learning is unnecessary - just adapt the classifier.

## How It Works

CAML uses a frozen, large-scale pretrained vision transformer as a universal feature extractor
and adapts to few-shot tasks using non-parametric, context-aware classification.

Reference: Fifty, C., Duan, D., Junkins, R.G., Amid, E., Leskovec, J.,
Re, C., & Thrun, S. (2023). Context-Aware Meta-Learning. NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CAMLOptions(IFullModel<,,>)` | Initializes a new instance of CAMLOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `ContextDimension` | Gets or sets the context embedding dimension. |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `FreezeBackbone` | Gets or sets whether to freeze the backbone during meta-training. |
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
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

