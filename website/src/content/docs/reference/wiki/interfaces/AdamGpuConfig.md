---
title: "AdamGpuConfig"
description: "Configuration for Adam optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for Adam optimizer on GPU.

## For Beginners

Adam is one of the most popular optimizers.
It adapts the learning rate for each parameter based on:

- First moment (mean of gradients) - like momentum
- Second moment (variance of gradients) - adapts to gradient magnitude

This typically leads to faster convergence than plain SGD.

## How It Works

Adam maintains moving averages of gradients (m) and squared gradients (v),
with bias correction for the initial steps.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdamGpuConfig(Single,Single,Single,Single,Single,Int32)` | Creates a new Adam GPU configuration. |

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

