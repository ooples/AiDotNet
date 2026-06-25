---
title: "LARSOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm.

## For Beginners

When training with large batches (common in self-supervised learning),
regular optimizers can become unstable. LARS solves this by automatically adjusting learning rates
for each layer based on how "big" the weights and gradients are. This makes training more stable
and allows you to use much larger batch sizes, which speeds up training significantly.

## How It Works

LARS (Layer-wise Adaptive Rate Scaling) is designed for training with very large batch sizes
(4096-32768). It automatically adapts the learning rate for each layer based on the ratio
of parameter norm to gradient norm, which helps maintain stable training with large batches.

LARS is particularly important for self-supervised learning methods like SimCLR, which achieve
their best results with batch sizes of 4096-8192.

Based on the paper "Large Batch Training of Convolutional Networks" by You et al. (2017).

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Epsilon` | Gets or sets a small constant added to denominators to prevent division by zero. |
| `ExcludeBiasFromLARS` | Gets or sets whether to exclude bias parameters and normalization layer parameters from LARS scaling. |
| `InitialLearningRate` | Gets or sets the base learning rate for the LARS optimizer. |
| `LayerBoundaries` | Gets or sets the layer size boundaries for layer-wise scaling. |
| `Momentum` | Gets or sets the momentum coefficient for the optimizer. |
| `SkipLARSLayers` | Gets or sets which layers should skip LARS scaling and use only the base learning rate. |
| `TrustCoefficient` | Gets or sets the LARS trust coefficient (eta). |
| `UseNesterov` | Gets or sets whether to use Nesterov momentum instead of standard momentum. |
| `WarmupEpochs` | Gets or sets the number of warmup steps for learning rate warmup. |
| `WeightDecay` | Gets or sets the weight decay coefficient. |

