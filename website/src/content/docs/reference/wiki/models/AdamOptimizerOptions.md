---
title: "AdamOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Adam optimization algorithm, which combines the benefits of AdaGrad and RMSProp."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Adam optimization algorithm, which combines the benefits of AdaGrad and RMSProp.

## For Beginners

Adam is like a smart learning assistant that remembers both the direction (momentum) and the
size of previous steps. It automatically adjusts how big each step should be for each parameter, making it easier to train
models without having to manually tune the learning rate. Adam is often a good default choice for many machine learning problems.

## How It Works

Adam (Adaptive Moment Estimation) is a popular optimization algorithm that computes adaptive learning rates for each parameter.
It stores both an exponentially decaying average of past gradients (first moment) and past squared gradients (second moment).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdamOptimizerOptions` | Default ctor — overrides the base `EnableGradientClipping` default from `false` to `true` with `MaxGradientNorm` at the canonical PyTorch transformer-training value of `1.0`. |
| `AdamOptimizerOptions(AdamOptimizerOptions<,,>)` | Copy constructor — clones every property declared on this options class plus the base-class gradient-clipping settings used here. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AnomalyGuardMode` | Gets or sets the policy for the PyTorch GradScaler-style anomaly guard that skips an Adam step when any gradient contains NaN or Inf. |
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta1` | Gets or sets the exponential decay rate for the first moment estimates. |
| `Beta2` | Gets or sets the exponential decay rate for the second moment estimates. |
| `Epsilon` | Gets or sets a small constant added to denominators to prevent division by zero. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the Adam optimizer. |
| `MaxBeta1` | Gets or sets the maximum allowed value for Beta1. |
| `MaxBeta2` | Gets or sets the maximum allowed value for Beta2. |
| `MinBeta1` | Gets or sets the minimum allowed value for Beta1. |
| `MinBeta2` | Gets or sets the minimum allowed value for Beta2. |
| `UseAMSGrad` | Gets or sets whether to use the AMSGrad variant of Adam. |
| `UseAdaptiveBetas` | Gets or sets whether to automatically adjust the Beta parameters during training. |

