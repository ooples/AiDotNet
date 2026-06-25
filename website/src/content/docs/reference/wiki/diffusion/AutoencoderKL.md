---
title: "AutoencoderKL<T>"
description: "KL-regularized Variational Autoencoder for latent diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.VAE`

KL-regularized Variational Autoencoder for latent diffusion models.

## For Beginners

AutoencoderKL is the "image compressor" used by Stable Diffusion.

Why use KL-regularized VAE?

1. Compression: 512x512x3 image -> 64x64x4 latent (48x smaller!)
2. KL-regularization: Keeps the latent space well-organized (Gaussian distribution)
3. This organization makes diffusion work better in latent space

The "KL" in AutoencoderKL refers to Kullback-Leibler divergence, which measures
how different the encoder's output distribution is from a standard normal.
By minimizing KL divergence, we ensure the latent space is smooth and continuous.

Architecture:
```
Image (512x512x3)
│
├─→ VAEEncoder ─→ [mean, logvar] (64x64x8)
│ │
│ Sample using reparameterization
│ │
│ ↓
│ Latent z (64x64x4)
│ │
│ [Scale by 0.18215]
│ │
│ ↓
│ Scaled latent (for diffusion)
│ │
│ [Unscale by 1/0.18215]
│ │
│ ↓
│ Latent z (64x64x4)
│ │
└────────────────→ VAEDecoder
│
↓
Reconstructed Image (512x512x3)
```

## How It Works

AutoencoderKL is the standard VAE architecture used in Stable Diffusion and other
latent diffusion models. It compresses high-resolution images to a compact latent
representation while maintaining perceptual quality through KL-regularization.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoencoderKL(Int32,Int32,Int32,Int32[],Int32,Int32,Nullable<Double>,Int32,ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the AutoencoderKL class with default Stable Diffusion configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Decoder` | Gets the decoder component for direct access. |
| `DownsampleFactor` |  |
| `Encoder` | Gets the encoder component for direct access. |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsSlicing` |  |
| `SupportsTiling` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackpropagateLossGradient(Tensor<>)` |  |
| `Clone` |  |
| `ComputeVAELoss(Tensor<>,Tensor<>,Double)` | Computes the VAE loss (reconstruction + KL divergence). |
| `Decode(Tensor<>)` |  |
| `DecodeFromDiffusion(Tensor<>)` | Decodes a diffusion latent back to image space. |
| `DeepCopy` |  |
| `Dispose(Boolean)` |  |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeForDiffusion(Tensor<>,Boolean)` | Encodes an image and applies latent scaling for use in diffusion. |
| `EncodeWithDistribution(Tensor<>)` |  |
| `Forward(Tensor<>)` | Performs a full forward pass: encode -> sample -> decode. |
| `GetParameters` |  |
| `InvalidateCompiledHosts` | Bumps the structure version on both per-direction `CompiledModelHost`s so the next Encode / Decode call drops the cached compiled plan and re-traces. |
| `Lightweight` | Creates a lightweight AutoencoderKL for testing/experimentation. |
| `LoadState(Stream)` |  |
| `ResetState` | Resets the internal state of encoder and decoder. |
| `SDXL` | Creates an AutoencoderKL matching SDXL configuration. |
| `SaveState(Stream)` |  |
| `SetParameters(Vector<>)` |  |
| `StableDiffusionV1` | Creates a default AutoencoderKL matching Stable Diffusion v1.5 configuration. |
| `Train(Tensor<>,Tensor<>)` | Trains the VAE on a single image. |

## Fields

| Field | Summary |
|:-----|:--------|
| `SD_LATENT_SCALE` | Standard Stable Diffusion latent scale factor. |
| `_baseChannels` | Base channel count for encoder/decoder. |
| `_cachedLogVar` | Cached log variance from last encoding. |
| `_cachedMean` | Cached mean from last encoding. |
| `_channelMults` | Channel multipliers for each level. |
| `_decoder` | The decoder component. |
| `_encoder` | The encoder component. |
| `_encoderHost` | Per-direction compiled-plan host. |
| `_inputChannels` | Number of input/output image channels. |
| `_latentChannels` | Number of latent channels. |
| `_latentScaleFactor` | Latent scale factor. |
| `_structureVersion` | Structure version bumped when the encoder / decoder layer graph mutates (e.g. |

