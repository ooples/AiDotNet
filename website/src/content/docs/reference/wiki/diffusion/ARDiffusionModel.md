---
title: "ARDiffusionModel<T>"
description: "AR-Diffusion model combining autoregressive and diffusion generation in latent space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

AR-Diffusion model combining autoregressive and diffusion generation in latent space.

## For Beginners

AR-Diffusion generates images like reading a book — left to
right, top to bottom. But instead of picking discrete tokens (like words), it uses
diffusion to generate continuous patches. This gives it the best of both worlds:
the structured generation of autoregressive models and the smooth outputs of diffusion.

## How It Works

AR-Diffusion generates image tokens autoregressively (left-to-right, top-to-bottom)
but applies diffusion to each token's continuous representation. This combines the
sequential coherence of autoregressive models with the continuous-space flexibility
of diffusion, enabling high-quality generation with natural spatial ordering.

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

