---
title: "AdamWOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the AdamW optimization algorithm with decoupled weight decay."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the AdamW optimization algorithm with decoupled weight decay.

## For Beginners

AdamW is an improved version of Adam that handles weight decay (a technique
to prevent overfitting) in a mathematically cleaner way. The difference might seem subtle, but AdamW
consistently achieves better results than Adam with L2 regularization, especially when training
large models like transformers. If you're not sure which to use, AdamW is generally the better choice.

## How It Works

AdamW (Adam with decoupled Weight decay) differs from Adam with L2 regularization.
In Adam with L2, weight decay is applied to the gradient before the adaptive learning rate
is computed. In AdamW, weight decay is applied directly to the weights after the Adam update,
which has been shown to improve generalization.

Based on the paper "Decoupled Weight Decay Regularization" by Ilya Loshchilov and Frank Hutter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdamWOptimizerOptions` | Default ctor — overrides the base `EnableGradientClipping` default from `false` to `true` with `MaxGradientNorm` at the canonical transformer-training value of `1.0`. |
| `AdamWOptimizerOptions(AdamWOptimizerOptions<,,>)` | Copy constructor — required by the Options golden pattern so Clone() faithfully preserves every property. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta1` | Gets or sets the exponential decay rate for the first moment estimates (momentum). |
| `Beta2` | Gets or sets the exponential decay rate for the second moment estimates (adaptive learning rate). |
| `Epsilon` | Gets or sets a small constant added to denominators to prevent division by zero. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the AdamW optimizer. |
| `MaxBeta1` | Gets or sets the maximum allowed value for Beta1. |
| `MaxBeta2` | Gets or sets the maximum allowed value for Beta2. |
| `MinBeta1` | Gets or sets the minimum allowed value for Beta1. |
| `MinBeta2` | Gets or sets the minimum allowed value for Beta2. |
| `UseAMSGrad` | Gets or sets whether to apply AMSGrad variant for improved convergence guarantees. |
| `UseAdaptiveBetas` | Gets or sets whether to automatically adjust the Beta parameters during training. |
| `WeightDecay` | Gets or sets the weight decay coefficient (L2 penalty). |

