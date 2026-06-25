---
title: "SymmetricProjector<T>"
description: "Symmetric Projector Head for BYOL and SimSiam-style methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Symmetric Projector Head for BYOL and SimSiam-style methods.

## For Beginners

The symmetric projector is used in BYOL and SimSiam.
It consists of a projector MLP followed by a predictor MLP. The predictor
creates asymmetry between online and target branches, which is key to avoiding collapse.

## How It Works

**Architecture:**

**Key insight:** The predictor is only applied to the online branch,
creating asymmetry. The target branch only uses the projector.

**Dual-branch caching:** This projector supports two concurrent forward contexts
(branch 1 and branch 2) so that symmetric multi-view training (BYOL, SimSiam, BarlowTwins)
can call Project() twice and then Backward() twice without the second forward overwriting
the first branch's cached activations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SymmetricProjector(Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the SymmetricProjector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasPredictor` | Gets whether this projector has a predictor head. |
| `HiddenDimension` |  |
| `InputDimension` |  |
| `OutputDimension` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchNormBackward(Tensor<>,[],[],Tensor<>,[],[])` | Full BatchNorm backward pass computing gradients for input, gamma, and beta. |
| `ClearGradients` |  |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `Predict(Tensor<>)` | Applies the predictor head using the most recently used branch from `Tensor{`. |
| `Predict(Tensor<>,Int32)` | Applies the predictor head (for online branch only). |
| `Project(Tensor<>)` |  |
| `ProjectAndPredict(Tensor<>)` | Projects and predicts in one call (convenience method). |
| `Reset` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` |  |

