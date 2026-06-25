---
title: "TemporalConvolution<T>"
description: "1D temporal convolution layer for video diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

1D temporal convolution layer for video diffusion models.

## For Beginners

Temporal Convolution applies 1D convolution along the time dimension, treating each spatial position independently. This pseudo-3D approach is much faster than full 3D attention while still modeling temporal relationships.

## How It Works

**References:**

- Paper: "Make-A-Video: Text-to-Video Generation without Text-Video Data" (Singer et al., 2022)
- Paper: "Video Diffusion Models" (Ho et al., 2022)

Temporal convolution applies 1D convolution across the time dimension for each spatial position.
This provides local temporal modeling (mixing information from adjacent frames) as a complement
to temporal attention (which provides global temporal modeling). Temporal convolutions are:

- More efficient than attention for short-range temporal dependencies
- Often used alongside temporal attention in video UNets
- Optionally causal (only looking at past frames) for streaming generation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalConvolution(Int32,Int32,Int32,Boolean)` | Initializes a new temporal convolution layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `IsCausal` | Gets whether causal convolution is used. |
| `KernelSize` | Gets the temporal kernel size. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Applies temporal convolution across frames. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

