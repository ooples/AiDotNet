---
title: "FEATAlgorithm<T, TInput, TOutput>"
description: "Implementation of FEAT (Few-shot Embedding Adaptation with Transformer)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of FEAT (Few-shot Embedding Adaptation with Transformer).

## For Beginners

FEAT makes prototypes smarter by letting them "see" each other:

**The insight:**
ProtoNets computes each class prototype in isolation. But if you know that class A
and class B are very similar, you should push their prototypes apart to avoid confusion.
FEAT uses a transformer to automatically learn these adjustments.

**How it works:**

1. Compute initial prototypes (mean of support features per class, like ProtoNets)
2. Feed ALL prototypes through a transformer
- The transformer uses self-attention so each prototype can "see" all others
- It learns to adjust prototypes based on the specific set of classes
3. Use the adapted prototypes for nearest-prototype classification
4. Train with both classification loss AND contrastive loss

**Why the transformer helps:**

- In a 5-way task with dogs vs cats: Prototypes need to capture species differences
- In a 5-way task with dog breeds: Same dog features need to capture breed differences
- The transformer adjusts prototypes based on what's needed for THIS specific task

## How It Works

FEAT uses a set-to-set transformer to adapt class prototypes based on inter-class
relationships. Initial prototypes (class means) are fed through the transformer,
which outputs task-adapted prototypes that are more discriminative.

**Algorithm - FEAT:**

Reference: Ye, H.J., Hu, H., Zhan, D.C., & Sha, F. (2020).
Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FEATAlgorithm(FEATOptions<,,>)` | Initializes a new FEAT meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using transformer-adapted prototypes. |
| `AdaptPrototypes(Vector<>)` | Adapts prototypes using the set-to-set transformer. |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using the transformer-adapted prototypes. |
| `ComputeModulationFactor(Vector<>,Vector<>)` | Computes a scalar modulation factor by comparing adapted vs raw feature vectors. |
| `InitializeTransformer` | Initializes the set-to-set transformer parameters. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for FEAT. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_transformerParams` | Parameters for the set-to-set transformer that adapts prototypes. |

