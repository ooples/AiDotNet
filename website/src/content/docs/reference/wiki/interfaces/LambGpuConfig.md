---
title: "LambGpuConfig"
description: "Configuration for LAMB (Layer-wise Adaptive Moments) optimizer on GPU."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Interfaces`

Configuration for LAMB (Layer-wise Adaptive Moments) optimizer on GPU.

## For Beginners

LAMB is like a combination of Adam and LARS.
It uses Adam's moment estimates for adaptive learning AND applies layer-wise
scaling like LARS. This enables training very large models (like BERT) with
very large batch sizes efficiently.

## How It Works

LAMB combines Adam's adaptive learning with LARS's layer-wise scaling.
It was designed for training BERT and other large transformers with huge batches.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LambGpuConfig(Single,Single,Single,Single,Single,Int32)` | Creates a new LAMB GPU configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Beta1` | Gets the exponential decay rate for the first moment estimates (typically 0.9). |
| `Beta2` | Gets the exponential decay rate for the second moment estimates (typically 0.999). |
| `Epsilon` | Gets the small constant for numerical stability (typically 1e-6). |
| `LearningRate` |  |
| `OptimizerType` |  |
| `Step` |  |
| `WeightDecay` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyUpdate(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,GpuOptimizerState,Int32)` |  |

