---
title: "SgdGpuConfig"
description: "Configuration for SGD (Stochastic Gradient Descent) optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for SGD (Stochastic Gradient Descent) optimizer on GPU.

## For Beginners

SGD is the simplest optimizer. It moves weights
in the direction opposite to the gradient, scaled by the learning rate.
Momentum helps accelerate training by accumulating velocity from past updates.

## How It Works

SGD updates weights using: w = w - lr * (grad + weightDecay * w) + momentum * velocity

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SgdGpuConfig(Single,Single,Single,Int32)` | Creates a new SGD GPU configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` |  |
| `Momentum` | Gets the momentum coefficient (typically 0.9). |
| `OptimizerType` |  |
| `Step` |  |
| `WeightDecay` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyUpdate(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,GpuOptimizerState,Int32)` |  |

