---
title: "AdamWGpuConfig"
description: "Configuration for AdamW optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for AdamW optimizer on GPU.

## For Beginners

AdamW fixes a subtle issue with L2 regularization in Adam.
The original Adam with weight decay doesn't properly regularize because the adaptive
learning rates interfere. AdamW applies weight decay directly to weights, which works better.

## How It Works

AdamW is Adam with decoupled weight decay. Instead of adding weight decay to the gradient
before the Adam update, it subtracts it directly from the weights after the update.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdamWGpuConfig(Single,Single,Single,Single,Single,Int32)` | Creates a new AdamW GPU configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta1` | Gets the exponential decay rate for the first moment estimates (typically 0.9). |
| `Beta2` | Gets the exponential decay rate for the second moment estimates (typically 0.999). |
| `Epsilon` | Gets the small constant for numerical stability (typically 1e-8). |
| `LearningRate` |  |
| `OptimizerType` |  |
| `Step` |  |
| `WeightDecay` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyUpdate(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,GpuOptimizerState,Int32)` |  |

