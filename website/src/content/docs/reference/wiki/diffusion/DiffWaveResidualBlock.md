---
title: "DiffWaveResidualBlock<T>"
description: "Residual block for DiffWave with dilated convolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

Residual block for DiffWave with dilated convolution.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffWaveResidualBlock(IEngine,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new residual block. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the number of parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BroadcastAddEmbedding(Tensor<>,Tensor<>)` | Broadcasts a per-step embedding `embedding` across the time axis of `x` and adds it, FiLM-style. |
| `Forward(Tensor<>,Tensor<>,Tensor<>)` | Forward pass returning output and skip connection — Kong et al. |
| `GetParameters` | Gets all parameters. |
| `SetParameters(Vector<>)` | Sets all parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_engine` | Engine for tape-tracked tensor ops. |

