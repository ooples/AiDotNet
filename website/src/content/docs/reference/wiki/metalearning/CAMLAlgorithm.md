---
title: "CAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023).

## For Beginners

CAML is built on a simple but powerful insight:

**The insight:**
Modern pretrained models (CLIP, DINO) produce such good features that you don't
need to fine-tune them. Instead, learn a lightweight context module that adapts
how you USE the features for each specific task.

**How it works:**

1. Extract features using a frozen pretrained backbone (no gradient computation needed)
2. Compute class prototypes from support features (like ProtoNets)
3. Pass prototypes through a context module that adjusts them based on the task
4. Classify queries by distance to context-adapted prototypes

**Why freeze the backbone?**

- Much faster training (no backbone gradients)
- Avoids overfitting on small support sets
- Preserves the rich representations learned during pretraining
- Only the small context module needs to be meta-learned

## How It Works

CAML uses a frozen pretrained backbone with a lightweight context module that adapts
features based on the support set. Classification is performed by comparing query features
to context-adapted class prototypes.

**Algorithm - CAML:**

Reference: Fifty, C., Duan, D., Junkins, R.G., Amid, E., Leskovec, J.,
Re, C., & Thrun, S. (2023). Context-Aware Meta-Learning. NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CAMLAlgorithm(CAMLOptions<,,>)` | Initializes a new CAML meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ApplyContextModule(Vector<>)` | Applies the context module to adapt prototypes based on the support set context. |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using the context module. |
| `InitializeContextModule` | Initializes the context module parameters. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextParams` | Parameters for the lightweight context module. |

