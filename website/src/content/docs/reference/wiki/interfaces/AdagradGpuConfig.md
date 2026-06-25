---
title: "AdagradGpuConfig"
description: "Configuration for Adagrad optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for Adagrad optimizer on GPU.

## For Beginners

Adagrad is good for sparse data because it gives larger
updates to infrequent parameters and smaller updates to frequent ones.
However, the accumulated squared gradients can make learning rate too small eventually.

## How It Works

Adagrad accumulates squared gradients over all time, providing automatic learning rate
adaptation. Parameters with frequently occurring features get smaller learning rates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdagradGpuConfig(Single,Single,Single,Int32)` | Creates a new Adagrad GPU configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Epsilon` | Gets the small constant for numerical stability (typically 1e-8). |
| `LearningRate` |  |
| `OptimizerType` |  |
| `Step` |  |
| `WeightDecay` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyUpdate(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,GpuOptimizerState,Int32)` |  |

