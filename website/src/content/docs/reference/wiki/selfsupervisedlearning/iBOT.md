---
title: "iBOT<T>"
description: "iBOT: Image BERT Pre-Training with Online Tokenizer - combining DINO with masked image modeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

iBOT: Image BERT Pre-Training with Online Tokenizer - combining DINO with masked image modeling.

## For Beginners

iBOT combines the best of DINO (self-distillation) with
masked image modeling (like MAE). It masks patches in the student view and predicts
both the CLS token (like DINO) and the masked patches (like BERT for images).

## How It Works

**Key innovations:**

**Loss formula:**

**Reference:** Zhou et al., "iBOT: Image BERT Pre-Training with Online Tokenizer"
(ICLR 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `iBOT(INeuralNetwork<>,IMomentumEncoder<>,IProjectorHead<>,IProjectorHead<>,Int32,Double,Double,SSLConfig)` | Initializes a new instance of the iBOT class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MIMWeight` | Gets the weight for masked image modeling loss. |
| `MaskRatio` | Gets the mask ratio for patches. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineLossGradients(Tensor<>,Tensor<>,Double,Double)` | Combines CLS and MIM loss gradients element-wise: result = clsWeight * gradCls + mimWeight * gradMim. |
| `ComputeMaskedPatchGradient(Tensor<>,Tensor<>,Tensor<>)` | Computes the gradient of the masked patch loss with respect to studentOut. |
| `Create(INeuralNetwork<>,Func<INeuralNetwork<>,INeuralNetwork<>>,Int32,Int32,Int32,Double,Double)` | Creates an iBOT instance with default configuration. |
| `UpdateStudent(,Vector<>)` | Updates the student network with pre-accumulated projector gradients. |

