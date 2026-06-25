---
title: "MixedPrecisionTrainingLoop<T>"
description: "Implements mixed-precision training loop for neural networks following NVIDIA's approach."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MixedPrecision`

Implements mixed-precision training loop for neural networks following NVIDIA's approach.

## For Beginners

This class implements the complete mixed-precision training workflow:

1. **Cast weights to FP16** - Convert FP32 master weights to FP16 working weights
2. **Forward pass in FP16** - Fast computation using 16-bit precision
3. **Compute loss in FP32** - Calculate error using 32-bit precision for stability
4. **Scale loss** - Multiply by large factor (e.g., 2^16) to prevent gradient underflow
5. **Backward pass in FP16** - Compute gradients in 16-bit precision
6. **Unscale and cast gradients to FP32** - Convert gradients back to 32-bit and divide by scale
7. **Check for overflow** - Detect NaN/Inf and adjust loss scale if needed
8. **Update master weights in FP32** - Apply gradients to 32-bit master weights

This workflow provides 2-3x speedup on modern GPUs while maintaining model accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixedPrecisionTrainingLoop(NeuralNetworkBase<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,MixedPrecisionContext,LayerPrecisionPolicy)` | Initializes a new mixed-precision training loop. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentLossScale` | Gets the current loss scale factor. |
| `LastLoss` | Gets the last computed loss value. |
| `Policy` | Gets the layer precision policy used by this training loop. |
| `SkippedSteps` | Gets the number of steps skipped due to gradient overflow. |
| `TotalSteps` | Gets the total number of training steps performed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDefaultPolicy(MixedPrecisionType)` | Gets the default layer precision policy based on the precision type. |
| `GetStatistics` | Gets statistics about the training process. |

