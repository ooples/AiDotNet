---
title: "LarsGpuConfig"
description: "Configuration for LARS (Layer-wise Adaptive Rate Scaling) optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for LARS (Layer-wise Adaptive Rate Scaling) optimizer on GPU.

## For Beginners

LARS was designed for training with huge batch sizes
(like 32K images). It automatically adjusts the learning rate for each layer
so that layers with large parameters don't update too fast and layers with
small parameters don't update too slow.

## How It Works

LARS scales the learning rate for each layer based on the ratio of parameter norm
to gradient norm. This enables training with very large batch sizes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LarsGpuConfig(Single,Single,Single,Single,Int32)` | Creates a new LARS GPU configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` |  |
| `Momentum` | Gets the momentum coefficient (typically 0.9). |
| `OptimizerType` |  |
| `Step` |  |
| `TrustCoefficient` | Gets the trust coefficient for layer-wise scaling (typically 0.001). |
| `WeightDecay` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyUpdate(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,GpuOptimizerState,Int32)` |  |

