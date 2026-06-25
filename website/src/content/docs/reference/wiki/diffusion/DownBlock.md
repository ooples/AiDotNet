---
title: "DownBlock<T>"
description: "Downsampling block for VAE encoder with multiple ResBlocks and strided convolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Downsampling block for VAE encoder with multiple ResBlocks and strided convolution.

## For Beginners

A DownBlock is like a compression stage in an encoder.

What it does:

1. Processes the input through multiple residual blocks (learning features)
2. Reduces spatial size by half using strided convolution (compression)

Example: 64x64 input -> 32x32 output (spatial dimensions halved)

Why use strided convolution instead of pooling?

- Strided conv is learnable (the network decides how to downsample)
- Max/Avg pooling has fixed behavior that may discard useful information
- Strided conv is the standard in modern generative models like VAEs and diffusion

Structure:
```
input [B, C_in, H, W]
│
├─→ ResBlock → ResBlock → ... (numLayers blocks)
│
↓
[B, C_out, H, W]
│
├─→ Conv3x3 (stride=2) ─→ downsample
│
↓
output [B, C_out, H/2, W/2]
```

## How It Works

This implements a downsampling block following the Stable Diffusion VAE architecture:

- Multiple VAEResBlocks to process features at the current resolution
- Strided convolution (stride=2) to reduce spatial dimensions by half

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DownBlock(Int32,Int32,Int32,Int32,Int32,Boolean)` | Initializes a new instance of the DownBlock class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasDownsample` | Gets whether this block performs downsampling. |
| `InputChannels` | Gets the number of input channels. |
| `NumLayers` | Gets the number of residual blocks. |
| `OutputChannels` | Gets the number of output channels. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(BinaryReader)` | Loads the block's state from a binary reader. |
| `Forward(Tensor<>)` | Performs the forward pass through the down block. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetResBlocks` | Gets the residual blocks for external access (e.g., for skip connections in UNet). |
| `ResetState` | Resets the internal state of the block. |
| `Serialize(BinaryWriter)` | Saves the block's state to a binary writer. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `UpdateParameters()` | Updates all learnable parameters using gradient descent. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_downsample` | Strided convolution for downsampling. |
| `_hasDownsample` | Whether this block includes downsampling (false for the last encoder block). |
| `_inChannels` | Number of input channels. |
| `_inputSpatialSize` | Spatial size at input. |
| `_lastInput` | Cached inputs and intermediate values for backward pass. |
| `_numGroups` | Number of groups for GroupNorm in ResBlocks. |
| `_numLayers` | Number of residual blocks. |
| `_outChannels` | Number of output channels. |
| `_resBlocks` | Residual blocks in this down block. |

