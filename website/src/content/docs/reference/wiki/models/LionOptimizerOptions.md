---
title: "LionOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Lion (Evolved Sign Momentum) optimization algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Lion (Evolved Sign Momentum) optimization algorithm.

## For Beginners

Think of Lion as a streamlined version of Adam that focuses on the direction
of learning (not the magnitude). It's like a compass that only tells you which way to go, making decisions
faster and using less memory. Lion is particularly effective for training large neural networks and transformers,
where it can achieve better results than Adam while using half the memory.

## How It Works

Lion is a modern optimization algorithm discovered through symbolic program search that offers
significant advantages over Adam, including 50% memory reduction and superior performance on large models.
Unlike Adam which maintains both momentum and variance, Lion uses only a single momentum state and
relies on sign-based updates for improved efficiency and generalization.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta1` | Gets or sets the exponential decay rate for the momentum interpolation (used for computing the update). |
| `Beta1DecreaseFactor` | Gets or sets the factor by which Beta1 is decreased when fitness does not improve. |
| `Beta1IncreaseFactor` | Gets or sets the factor by which Beta1 is increased when fitness improves. |
| `Beta2` | Gets or sets the exponential decay rate for updating the momentum state. |
| `Beta2DecreaseFactor` | Gets or sets the factor by which Beta2 is decreased when fitness does not improve. |
| `Beta2IncreaseFactor` | Gets or sets the factor by which Beta2 is increased when fitness improves. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the Lion optimizer. |
| `MaxBeta1` | Gets or sets the maximum allowed value for Beta1. |
| `MaxBeta2` | Gets or sets the maximum allowed value for Beta2. |
| `MinBeta1` | Gets or sets the minimum allowed value for Beta1. |
| `MinBeta2` | Gets or sets the minimum allowed value for Beta2. |
| `UseAdaptiveBeta1` | Gets or sets whether to automatically adjust Beta1 during training. |
| `UseAdaptiveBeta2` | Gets or sets whether to automatically adjust Beta2 during training. |
| `WeightDecay` | Gets or sets the weight decay (L2 regularization) coefficient. |

