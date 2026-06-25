---
title: "DiffusionAttention<T>"
description: "Memory-efficient attention layer for diffusion models using Flash Attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

Memory-efficient attention layer for diffusion models using Flash Attention.

## For Beginners

Attention in diffusion models is computationally expensive.

For a 512x512 image at 8x downsampling:

- Sequence length = (512/8)^2 = 4096 tokens
- Standard attention: 4096 x 4096 = 16 million attention weights!

This class automatically uses Flash Attention for long sequences:

- Under 256 tokens: Standard attention (faster for short sequences)
- 256+ tokens: Flash Attention (memory-efficient, scales better)

Usage:
```cs
var attention = new DiffusionAttention<float>(
channels: 320,
numHeads: 8,
spatialSize: 64);

var output = attention.Forward(input);
```

## How It Works

This attention layer automatically uses Flash Attention when the sequence length
exceeds a threshold, providing significant memory and performance benefits for
high-resolution image generation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionAttention(Int32,Int32,Int32,Int32,Boolean)` | Initializes a new diffusion attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `FlashAttentionEnabled` | Gets whether Flash Attention is enabled. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` |  |
| `SupportsTraining` | Whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through the attention layer. |
| `GetDiagnostics` | Gets diagnostic information about the layer. |
| `GetParameters` | Gets all layer parameters as a single vector. |
| `ResetState` | Resets the layer's internal state. |
| `SetParameters(Vector<>)` | Sets all layer parameters from a single vector. |
| `SetTrainingMode(Boolean)` | Propagates eval/training mode to the nested Flash / standard attention sublayers. |
| `UpdateParameters()` | Updates parameters using computed gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_channels` | Number of channels. |
| `_flashAttention` | Flash Attention layer for long sequences. |
| `_flashAttentionThreshold` | Sequence length threshold for switching to Flash Attention. |
| `_flashConfig` | Flash Attention configuration. |
| `_headDim` | Dimension per head. |
| `_lastInput` | Cached input for backward pass. |
| `_numHeads` | Number of attention heads. |
| `_spatialSize` | Spatial size (height/width). |
| `_standardAttention` | Standard attention layer for short sequences. |
| `_usedFlashAttention` | Whether Flash Attention was used in the last forward pass. |

