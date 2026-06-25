---
title: "DiffusionResBlock<T>"
description: "Implements a residual block per the DDPM (Ho et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Implements a residual block per the DDPM (Ho et al. 2020) and Stable Diffusion (Rombach et al. 2022)
U-Net architecture with time embedding conditioning.

## How It Works

The forward pass implements the following computation:

where `skip_conv` is a 1x1 convolution if `inChannels != outChannels`, otherwise identity.

Performance: all intermediate tensors use `TensorAllocator` for pooled allocation.
GroupNorm uses 32 groups (SD standard) with channels that aren't divisible by 32 falling back
to the largest divisor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionResBlock(Int32,Int32,Int32,Int32,Int32)` | Creates a new diffusion residual block per the DDPM/Stable Diffusion paper. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySiLUBackward(Tensor<>,Tensor<>)` | SiLU/Swish backward: d/dx[x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)) = sigmoid(x) * (1 + x*(1-sigmoid(x))) |
| `ComputeNumGroups(Int32,Int32)` | Computes appropriate number of groups for GroupNorm. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Forward pass implementing the DDPM residual block. |
| `Forward(Tensor<>,Tensor<>)` | Forward pass with time embedding conditioning per the DDPM paper. |
| `GetParameters` |  |
| `GetTimeEmbedGradient` | Gets the accumulated time embedding gradient from the last backward pass. |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` | Propagates eval/training mode to the block's nested sublayers. |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputPortsCache` | Declares named input ports. |

