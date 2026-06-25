---
title: "IVAEModel<T>"
description: "Interface for Variational Autoencoder (VAE) models used in latent diffusion."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for Variational Autoencoder (VAE) models used in latent diffusion.

## For Beginners

A VAE is like a very smart image compressor and decompressor.

How it works:

1. Encoder: Takes a full-size image (e.g., 512x512x3) and compresses it to a small latent (e.g., 64x64x4)
2. Decoder: Takes the small latent and reconstructs a full-size image
3. The compression is lossy but learned to preserve important visual information

Why use a VAE in diffusion?

- Full images are huge (512x512x3 = 786,432 values)
- Latents are small (64x64x4 = 16,384 values) - 48x smaller!
- Diffusion in latent space is much faster
- Quality remains high because the VAE learns what matters

Different VAE types:

- Standard VAE: Original Stable Diffusion VAE, 4 latent channels
- Tiny VAE: Faster but lower quality, good for previews
- Temporal VAE: Video-aware VAE that handles frame consistency

## How It Works

VAEs are used in latent diffusion models to compress images into a lower-dimensional
latent space where the diffusion process operates. This makes training and generation
much more efficient than operating in pixel space.

This interface extends `IFullModel` to provide all standard
model capabilities (training, saving, loading, gradients, checkpointing, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `DownsampleFactor` | Gets the spatial downsampling factor. |
| `InputChannels` | Gets the number of input channels (image channels). |
| `LatentChannels` | Gets the number of latent channels. |
| `LatentScaleFactor` | Gets the scale factor for latent values. |
| `SupportsSlicing` | Gets whether this VAE uses slicing for sequential processing. |
| `SupportsTiling` | Gets whether this VAE uses tiling for memory-efficient encoding/decoding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(Tensor<>)` | Decodes a latent representation back to image space. |
| `Encode(Tensor<>,Boolean)` | Encodes an image into the latent space. |
| `EncodeWithDistribution(Tensor<>)` | Encodes and returns both mean and log variance (for training). |
| `Sample(Tensor<>,Tensor<>,Nullable<Int32>)` | Samples from the latent distribution using the reparameterization trick. |
| `ScaleLatent(Tensor<>)` | Scales latent values for use in diffusion (applies LatentScaleFactor). |
| `SetSlicingEnabled(Boolean)` | Enables or disables slicing mode. |
| `SetTilingEnabled(Boolean)` | Enables or disables tiling mode. |
| `UnscaleLatent(Tensor<>)` | Unscales latent values before decoding (inverts LatentScaleFactor). |

