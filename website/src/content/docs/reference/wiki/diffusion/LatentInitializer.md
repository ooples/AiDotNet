---
title: "LatentInitializer<T>"
description: "Initializes latent tensors for diffusion generation with various noise strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Initializes latent tensors for diffusion generation with various noise strategies.

## For Beginners

Before a diffusion model can create an image, it needs a starting
point — usually pure random noise. LatentInitializer handles creating this starting noise.
For text-to-image, it creates pure noise. For image editing (img2img), it mixes the original
image with noise, controlling how much of the original to keep.

## How It Works

Provides different strategies for initializing the starting noise latent for diffusion
generation. Supports standard Gaussian noise, image-conditioned initialization (img2img),
and strength-based partial noise for editing workflows. Ensures consistent shapes and
proper scaling for the target scheduler.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LatentInitializer(Double,Nullable<Int32>)` | Initializes a new latent initializer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InitNoiseSigma` | Gets the initial noise sigma scaling factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateImg2ImgLatent(Vector<>,Double)` | Creates a latent initialized from an encoded image with noise added (img2img). |
| `CreateNoiseLatent(Int32)` | Creates a pure noise latent for text-to-image generation. |

