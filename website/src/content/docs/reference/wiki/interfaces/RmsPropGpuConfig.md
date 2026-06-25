---
title: "RmsPropGpuConfig"
description: "Configuration for RMSprop optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for RMSprop optimizer on GPU.

## For Beginners

RMSprop adapts the learning rate by dividing by a running
average of gradient magnitudes. This helps training be more stable when gradients
vary a lot in size - common in recurrent neural networks.

## How It Works

RMSprop maintains a moving average of squared gradients to normalize the gradient.
This helps with non-stationary objectives and is particularly useful for RNNs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RmsPropGpuConfig(Single,Single,Single,Single,Int32)` | Creates a new RMSprop GPU configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Epsilon` | Gets the small constant for numerical stability (typically 1e-8). |
| `LearningRate` |  |
| `OptimizerType` |  |
| `Rho` | Gets the decay rate for the moving average (typically 0.9). |
| `Step` |  |
| `WeightDecay` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyUpdate(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,GpuOptimizerState,Int32)` |  |

