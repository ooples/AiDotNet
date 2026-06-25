---
title: "NagGpuConfig"
description: "Configuration for Nesterov Accelerated Gradient (NAG) optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for Nesterov Accelerated Gradient (NAG) optimizer on GPU.

## For Beginners

NAG improves on regular momentum by being smarter
about where to look. Instead of computing the gradient at the current position,
it first moves in the direction of accumulated momentum, then computes the gradient.
This "lookahead" helps it slow down before overshooting.

## How It Works

NAG is a variation of momentum that looks ahead by evaluating the gradient
at the "lookahead" position. This often leads to better convergence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NagGpuConfig(Single,Single,Single,Int32)` | Creates a new NAG GPU configuration. |

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

