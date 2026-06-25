---
title: "LossScaler<T>"
description: "Implements dynamic loss scaling for mixed-precision training to prevent gradient underflow."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MixedPrecision`

Implements dynamic loss scaling for mixed-precision training to prevent gradient underflow.

## For Beginners

Loss scaling is a technique used in mixed-precision training to prevent very small
gradient values from becoming zero (underflow) when using 16-bit precision.

The problem:

- FP16 (Half) can only represent numbers in the range [6e-8, 65504]
- During training, gradients are often very small (e.g., 1e-10)
- Small gradients underflow to zero in FP16, stopping learning

The solution:

- Scale the loss by a large factor (e.g., 2^16 = 65536) before backpropagation
- This makes gradients larger, preventing underflow
- Unscale gradients back to their original values before parameter updates

Dynamic scaling:

- Automatically adjusts the scale factor during training
- Increases scale when gradients are stable (no overflow)
- Decreases scale when gradients overflow (become infinity/NaN)

## How It Works

**Technical Details:** The algorithm follows NVIDIA's approach:

1. Start with a large initial scale (default: 2^16 = 65536)
2. If no overflow for N steps, increase scale by growth factor (default: 2.0)
3. If overflow detected, decrease scale by backoff factor (default: 0.5) and skip update
4. Monitor consecutive successful updates for scale adjustment

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LossScaler(Double,Boolean,Int32,Double,Double,Double,Double)` | Initializes a new instance of the LossScaler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BackoffFactor` | Factor by which to multiply the scale when decreasing (default: 0.5). |
| `DynamicScaling` | Whether to use dynamic loss scaling. |
| `GrowthFactor` | Factor by which to multiply the scale when increasing (default: 2.0). |
| `GrowthInterval` | Number of consecutive iterations without overflow before increasing scale. |
| `MaxScale` | Maximum allowed scale value to prevent excessive growth. |
| `MinScale` | Minimum allowed scale value to prevent excessive reduction. |
| `OverflowRate` | Gets the overflow rate (skipped / total). |
| `Scale` | Current loss scale factor. |
| `SkippedUpdates` | Gets the number of updates skipped due to overflow. |
| `TotalUpdates` | Gets the total number of updates attempted. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectOverflow(Tensor<>)` | Checks if any gradient in a tensor has overflowed. |
| `DetectOverflow(Vector<>)` | Checks if any gradient in a vector has overflowed. |
| `HasOverflow()` | Checks if a single value has overflowed (is NaN or infinity). |
| `RecordOverflow` | Records that the current optimizer step OVERFLOWED — equivalent to `Tensor{` finding NaN/Inf in the scaled gradients, but without touching any gradient buffer. |
| `RecordSuccess` | Records that the current optimizer step succeeded — equivalent to `Tensor{` finding no NaN/Inf in the scaled gradients, but without touching any gradient buffer. |
| `Reset(Nullable<Double>)` | Resets the statistics and scale to initial values. |
| `ScaleLoss()` | Scales the loss value to prevent gradient underflow. |
| `ToString` | Gets a summary of the loss scaler's current state. |
| `UnscaleGradient()` | Unscales a single gradient value. |
| `UnscaleGradients(Tensor<>)` | Unscales all gradients in a tensor. |
| `UnscaleGradients(Vector<>)` | Unscales all gradients in a vector. |
| `UnscaleGradientsAndCheck(Tensor<>)` | Unscales gradients and checks for overflow, updating the scale factor if dynamic scaling is enabled. |
| `UnscaleGradientsAndCheck(Vector<>)` | Unscales gradients and checks for overflow (vector version). |

