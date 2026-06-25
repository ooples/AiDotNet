---
title: "StandardVAE<T>"
description: "Standard Variational Autoencoder for latent diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

Standard Variational Autoencoder for latent diffusion models.

## For Beginners

The StandardVAE is like a very smart image compressor:

How it works:

1. Encoder: Takes a 512x512x3 image and compresses it to 64x64x4 latent
- That's 48x compression! (786,432 values -> 16,384 values)
- Uses multiple layers of convolutions and downsampling

2. Decoder: Takes the 64x64x4 latent and reconstructs a 512x512x3 image
- Uses upsampling and convolutions to expand back to full size
- The reconstruction isn't perfect but preserves important visual features

Why 4 latent channels?

- The VAE learns to pack image information into 4 channels
- Each channel captures different aspects (colors, edges, textures, etc.)
- More channels = better quality but larger latent space

Why 8x downsampling?

- Each side is reduced by 8 (512 -> 64)
- This is the sweet spot between compression and quality
- Smaller latents = faster diffusion, but potentially lower quality

## How It Works

This implements a standard VAE architecture similar to Stable Diffusion's VAE,
with an encoder that compresses images to latent space and a decoder that
reconstructs images from latents.

Architecture details:

- Input: [batch, 3, H, W] RGB image normalized to [-1, 1]
- Encoder: ResBlocks with GroupNorm, downsampling via strided conv
- Latent: [batch, 4, H/8, W/8] with mean and variance for sampling
- Decoder: ResBlocks with GroupNorm, upsampling via transpose conv
- Output: [batch, 3, H, W] reconstructed image

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StandardVAE(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32[],Int32,Nullable<Double>,List<ILayer<>>,List<ILayer<>>,ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the StandardVAE class with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` |  |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsSlicing` |  |
| `SupportsTiling` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AssignDecoderLayers` | Assigns decoder layers from LayerHelper, extracting special layers by position. |
| `AssignEncoderLayers` | Assigns encoder layers from LayerHelper, extracting special layers by position. |
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `ComputeVAELoss(Tensor<>,Tensor<>,Double)` | Computes the VAE loss (reconstruction + KL divergence). |
| `Decode(Tensor<>)` |  |
| `DecodeFromDiffusion(Tensor<>)` | Decodes a diffusion latent back to image space. |
| `DeepCopy` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeForDiffusion(Tensor<>,Boolean)` | Encodes an image and applies latent scaling for use in diffusion. |
| `EncodeWithDistribution(Tensor<>)` |  |
| `GetParameterChunks` |  |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `InitializeLayers(NeuralNetworkArchitecture<>,List<ILayer<>>,List<ILayer<>>)` | Initializes encoder and decoder layers, using custom layers from the user if provided or creating industry-standard layers from the Stable Diffusion paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `SD_LATENT_SCALE` | Standard Stable Diffusion latent scale factor. |
| `_architecture` | The neural network architecture configuration, if provided. |
| `_baseChannels` | Base channel count. |
| `_cachedLogVar` | Cached log variance from encoding. |
| `_cachedMean` | Cached mean from encoding. |
| `_channelMultipliers` | Channel multipliers for each level. |
| `_decoderLayers` | Decoder layers. |
| `_downsampleFactor` | Downsampling factor. |
| `_encoderLayers` | Encoder layers. |
| `_inputChannels` | Input channels (3 for RGB). |
| `_inputConv` | Input convolution to initial embedding. |
| `_latentChannels` | Latent channels. |
| `_latentScaleFactor` | Latent scale factor. |
| `_layersInitialized` | Tracks whether the VAE layer graph has been built. |
| `_lazyShapesResolved` | Runs one dummy Encode + Decode pass through the network at a small spatial size so every lazy layer resolves its weight shapes. |
| `_logVarConv` | Log variance projection layer for latent distribution. |
| `_meanConv` | Mean projection layer for latent distribution. |
| `_numDecoderBlocks` | Number of decoder blocks. |
| `_numEncoderBlocks` | Number of encoder blocks. |
| `_numResBlocksPerLevel` | Number of residual blocks per level. |
| `_outputConv` | Output convolution to RGB. |
| `_postQuantConv` | Post-quant convolution in decoder. |
| `_preserveMaterializedParameters` | True once this VAE has runtime state that a clone must preserve. |
| `_quantConv` | Quant convolution from latent to decoder. |

