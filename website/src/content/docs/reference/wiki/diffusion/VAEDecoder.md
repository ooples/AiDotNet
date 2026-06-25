---
title: "VAEDecoder<T>"
description: "Convolutional decoder for VAE that reconstructs images from latent space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Convolutional decoder for VAE that reconstructs images from latent space.

## For Beginners

The VAE decoder is like an intelligent image decompressor.

What it does step by step:

1. Takes a compressed latent (e.g., 64x64x4)
2. Post-quant conv: Expands channels (4 -> 512)
3. Middle blocks: Extra processing at the bottleneck
4. UpBlocks: Progressively doubles resolution while decreasing channels
- Block 1: 512 channels, 64x64 -> 64x64 (no upsample at start)
- Block 2: 512 channels, 64x64 -> 128x128
- Block 3: 256 channels, 128x128 -> 256x256
- Block 4: 128 channels, 256x256 -> 512x512
5. Output: Produces 3-channel RGB image with tanh activation

The result is a high-resolution image reconstructed from the compressed latent.

## How It Works

This implements the decoder portion of a VAE following the Stable Diffusion architecture:

- Post-quant convolution to expand latent channels
- Middle blocks at the bottleneck
- Multiple UpBlocks with transposed conv upsampling and ResBlocks
- Output convolution to produce final image channels

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VAEDecoder(Int32,Int32,Int32,Int32[],Int32,Int32,Int32)` | Initializes a new instance of the VAEDecoder class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LatentChannels` | Gets the number of latent channels. |
| `OutputChannels` | Gets the number of output channels. |
| `SupportsTraining` |  |
| `UpsampleFactor` | Gets the upsampling factor (spatial expansion from input to output). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(BinaryReader)` | Loads the decoder's state from a binary reader. |
| `Forward(Tensor<>)` | Decodes a latent representation to an image. |
| `ForwardAsync(Tensor<>,CancellationToken)` | Async overload of `Tensor{` — routes through the compile host's `PredictAsync`. |
| `ForwardEager(Tensor<>)` | Eager forward pass body — invoked by the compile host on cache miss / when compilation is disabled, also reused by the backward path that depends on the cached intermediate tensors. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `InvalidateCompiledPlans` | Bumps the structure-version so the next Forward drops stale plans. |
| `ResetState` | Resets the internal state of the decoder. |
| `Serialize(BinaryWriter)` | Saves the decoder's state to a binary writer. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `UpdateParameters()` | Updates all learnable parameters using gradient descent. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseChannels` | Base channel count. |
| `_bottleneckSize` | Spatial size at decoder input (bottleneck). |
| `_channelMults` | Channel multipliers for each level. |
| `_compileHost` | Per-instance compile host. |
| `_inputConv` | Convolution to expand latent to decoder channels. |
| `_lastInput` | Cached intermediate values for backward pass. |
| `_latentChannels` | Number of latent channels. |
| `_midBlocks` | Middle residual blocks at the bottleneck. |
| `_normOut` | Group normalization before output. |
| `_numGroups` | Number of groups for GroupNorm. |
| `_numResBlocks` | Number of residual blocks per up-stage. |
| `_outputChannels` | Number of output image channels. |
| `_outputConv` | Output convolution to image channels. |
| `_outputSpatialSize` | Final output spatial size. |
| `_postQuantConv` | Post-quant convolution to expand latent channels. |
| `_silu` | SiLU activation function. |
| `_tanh` | Tanh activation for output. |
| `_upBlocks` | Upsampling blocks. |

