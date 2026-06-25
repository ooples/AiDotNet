---
title: "ILatentDiffusionModel<T>"
description: "Interface for latent diffusion models that operate in a compressed latent space."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for latent diffusion models that operate in a compressed latent space.

## For Beginners

Latent diffusion combines the power of diffusion models with the
efficiency of autoencoders.

How it works:

1. A VAE compresses images (512x512) into small latents (64x64)
2. Diffusion happens in this compressed space (much faster!)
3. The VAE decompresses the result back to a full image

Benefits:

- Training is ~50x faster than pixel-space diffusion
- Generation is ~50x faster
- Quality remains very high
- Enables practical high-resolution generation

Key components:

- VAE: Compresses and decompresses images
- Noise Predictor (U-Net/DiT): Predicts noise in latent space
- Scheduler: Controls the denoising process
- Conditioner: Encodes text/images for guided generation

## How It Works

Latent diffusion models are a highly efficient variant of diffusion models that perform
the denoising process in a compressed latent space rather than pixel space. This is the
architecture behind Stable Diffusion and many other state-of-the-art generative models.

This interface extends `IDiffusionModel` with latent-space specific operations.

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` | Gets the conditioning module (optional, for conditioned generation). |
| `GuidanceScale` | Gets the default guidance scale for classifier-free guidance. |
| `LatentChannels` | Gets the number of latent channels. |
| `NoisePredictor` | Gets the noise predictor model (U-Net, DiT, etc.). |
| `SupportsInpainting` | Gets whether this model supports inpainting. |
| `SupportsNegativePrompt` | Gets whether this model supports negative prompts. |
| `VAE` | Gets the VAE model used for encoding and decoding. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecodeFromLatent(Tensor<>)` | Decodes a latent representation back to an image. |
| `EncodeToLatent(Tensor<>,Boolean)` | Encodes an image into latent space. |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates images from text prompts using classifier-free guidance. |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` | Performs image-to-image generation (style transfer, editing). |
| `Inpaint(Tensor<>,Tensor<>,String,String,Int32,Nullable<Double>,Nullable<Int32>)` | Performs inpainting (filling in masked regions). |
| `SetGuidanceScale(Double)` | Sets the guidance scale for classifier-free guidance. |

