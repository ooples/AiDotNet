---
title: "FactorizedSpatioTemporalAttention<T>"
description: "Factorized spatio-temporal attention that applies spatial and temporal attention separately."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

Factorized spatio-temporal attention that applies spatial and temporal attention separately.

## For Beginners

Factorized Spatio-Temporal Attention processes spatial (within-frame) and temporal (across-frame) relationships separately. This is much more efficient than joint attention while still capturing both spatial detail and temporal motion.

## How It Works

**References:**

- Paper: "Scalable Diffusion Models with Transformers" (Peebles and Xie, 2023)
- Paper: "Video Diffusion Models" (Ho et al., 2022)

Factorized spatio-temporal attention decomposes full 3D attention into separate spatial
and temporal components. This reduces computational complexity from O((T*H*W)^2) to
O(T*(H*W)^2 + H*W*T^2), making it feasible for high-resolution long videos.

Architecture:

- Spatial attention: self-attention within each frame (across H*W positions)
- Temporal attention: self-attention across frames (for each spatial position)
- LayerNorm + residual connections around each attention block

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FactorizedSpatioTemporalAttention(Int32,Int32,Int32,Int32)` | Initializes a new factorized spatio-temporal attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `NumFrames` | Gets the number of frames. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Applies spatial attention then temporal attention with residual connections. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

