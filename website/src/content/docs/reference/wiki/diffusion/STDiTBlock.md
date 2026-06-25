---
title: "STDiTBlock<T>"
description: "Spatial-Temporal DiT (STDiT) block for efficient video generation transformers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

Spatial-Temporal DiT (STDiT) block for efficient video generation transformers.

## For Beginners

The STDiT (Spatial-Temporal DiT) Block alternates between spatial and temporal attention within a single transformer block. This efficient design is used in Open-Sora for balanced spatial and temporal processing.

## How It Works

**References:**

- Paper: "Open-Sora: Democratizing Efficient Video Production for All" (2024)
- Paper: "Latte: Latent Diffusion Transformer for Video Generation" (Ma et al., 2024)

STDiT is the core building block for video DiT architectures. It combines spatial and temporal
attention with cross-attention conditioning in a single transformer block:

1. Spatial self-attention (within each frame)
2. Temporal self-attention (across frames per spatial position)
3. Cross-attention to text conditioning
4. Feed-forward network

Each sub-layer uses adaptive layer normalization (adaLN-Zero) for timestep conditioning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STDiTBlock(Int32,Int32,Int32,Int32,Int32,Int32)` | Initializes a new STDiT block. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the hidden dimension. |
| `ContextDim` | Gets the context dimension for cross-attention. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Applies the STDiT block: spatial attn -> temporal attn -> cross attn -> FFN. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

