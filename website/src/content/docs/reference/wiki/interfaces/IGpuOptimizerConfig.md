---
title: "IGpuOptimizerConfig"
description: "Configuration for GPU-resident optimizer updates."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Configuration for GPU-resident optimizer updates.

## For Beginners

When training on GPU, the weights need to be updated
using an optimizer (like SGD or Adam). This configuration tells the GPU
exactly how to update the weights - with what learning rate, momentum, etc.

## How It Works

This interface allows layers to receive optimizer-specific configuration
for GPU parameter updates. Different optimizer types (SGD, Adam, etc.)
have different implementations with their specific hyperparameters.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets the learning rate for parameter updates. |
| `OptimizerType` | Gets the type of optimizer (SGD, Adam, AdamW, etc.). |
| `Step` | Gets the current optimization step (used for bias correction in Adam-family optimizers). |
| `WeightDecay` | Gets the weight decay (L2 regularization) coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyUpdate(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,GpuOptimizerState,Int32)` | Applies the optimizer update to the given parameter buffer using its gradient. |

