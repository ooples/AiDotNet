---
title: "LAMBOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm.

## For Beginners

LAMB is designed for training large models (like BERT, transformers)
with very large batch sizes. It combines:

- **From Adam:** Adaptive learning rates that adjust per-parameter based on gradient history
- **From LARS:** Layer-wise scaling that stabilizes large batch training

The result is an optimizer that can train at batch sizes of 16K-32K while achieving the same
accuracy as training with small batches, just much faster.

## How It Works

LAMB combines Adam's adaptive learning rates (first and second moment estimates) with LARS's
layer-wise trust ratio scaling. This enables training with extremely large batch sizes
(up to 32K) while maintaining training stability and accuracy.

Based on the paper "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
by You et al. (2019).

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Beta1` | Gets or sets the exponential decay rate for the first moment estimates (momentum). |
| `Beta2` | Gets or sets the exponential decay rate for the second moment estimates. |
| `ClipTrustRatio` | Gets or sets whether to clip the trust ratio to prevent extreme scaling. |
| `Epsilon` | Gets or sets a small constant added to denominators to prevent division by zero. |
| `ExcludeBiasFromWeightDecay` | Gets or sets whether to exclude bias and normalization parameters from weight decay. |
| `InitialLearningRate` | Gets or sets the base learning rate for the LAMB optimizer. |
| `LayerBoundaries` | Gets or sets the layer size boundaries for layer-wise scaling. |
| `MaxTrustRatio` | Gets or sets the maximum trust ratio when clipping is enabled. |
| `SkipTrustRatioLayers` | Gets or sets which layers should skip trust ratio scaling and use only Adam updates. |
| `UseBiasCorrection` | Gets or sets whether to use bias correction for the moment estimates. |
| `WarmupEpochs` | Gets or sets the number of warmup epochs for learning rate warmup. |
| `WeightDecay` | Gets or sets the weight decay coefficient. |

