---
title: "UpBlock<T>"
description: "Upsampling block for VAE decoder with transposed convolution and multiple ResBlocks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Upsampling block for VAE decoder with transposed convolution and multiple ResBlocks.

## For Beginners

An UpBlock is like a decompression stage in a decoder.

What it does:

1. Increases spatial size by 2x using transposed convolution (decompression)
2. Processes the upsampled features through multiple residual blocks

Example: 8x8 input -> 16x16 output (spatial dimensions doubled)

Why use transposed convolution instead of simple interpolation?

- Transposed conv is learnable (the network decides how to upsample)
- Simple interpolation (bilinear, nearest) has fixed behavior
- Learnable upsampling can generate sharper details

Structure:
```
input [B, C_in, H, W]
│
├─→ ConvTranspose (stride=2) ─→ upsample
│
↓
[B, C_out, 2*H, 2*W]
│
├─→ ResBlock → ResBlock → ... (numLayers blocks)
│
↓
output [B, C_out, 2*H, 2*W]
```

## How It Works

This implements an upsampling block following the Stable Diffusion VAE architecture:

- Transposed convolution (deconvolution) to increase spatial dimensions by 2x
- Multiple VAEResBlocks to process features at the upsampled resolution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UpBlock(Int32,Int32,Int32,Int32,Int32,Boolean)` | Initializes a new instance of the UpBlock class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasUpsample` | Gets whether this block performs upsampling. |
| `InputChannels` | Gets the number of input channels. |
| `NumLayers` | Gets the number of residual blocks. |
| `OutputChannels` | Gets the number of output channels. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(BinaryReader)` | Loads the block's state from a binary reader. |
| `Forward(Tensor<>)` | Performs the forward pass through the up block. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetResBlocks` | Gets the residual blocks for external access. |
| `ResetState` | Resets the internal state of the block. |
| `Serialize(BinaryWriter)` | Saves the block's state to a binary writer. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `UpdateParameters()` | Updates all learnable parameters using gradient descent. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hasUpsample` | Whether this block includes upsampling (false for the first decoder block). |
| `_inChannels` | Number of input channels. |
| `_inputSpatialSize` | Spatial size at input (before upsampling). |
| `_lastInput` | Cached inputs and intermediate values for backward pass. |
| `_numGroups` | Number of groups for GroupNorm in ResBlocks. |
| `_numLayers` | Number of residual blocks. |
| `_outChannels` | Number of output channels. |
| `_outputSpatialSize` | Spatial size at output (after upsampling). |
| `_resBlocks` | Residual blocks in this up block. |
| `_upsample` | Transposed convolution for upsampling. |

