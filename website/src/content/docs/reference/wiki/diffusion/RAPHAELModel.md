---
title: "RAPHAELModel<T>"
description: "RAPHAEL model — Mixture-of-Experts text-to-image diffusion model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

RAPHAEL model — Mixture-of-Experts text-to-image diffusion model.

## For Beginners

RAPHAEL uses specialized experts for different concepts:

How RAPHAEL works:

1. Text prompt is parsed into concept tokens
2. Each concept routes to specialized expert pathways in cross-attention
3. Space-MoE: different spatial regions use different experts
4. Time-MoE: different denoising steps use different experts
5. This enables precise control over each concept's visual appearance

Key characteristics:

- MoE cross-attention: each token gets specialized experts
- Space-MoE: experts specialize in spatial regions
- Time-MoE: experts specialize in denoising stages
- Better text-image alignment than standard U-Net

Use RAPHAEL when you need:

- Precise text-to-image alignment
- Multi-concept scenes with distinct attributes
- High aesthetic quality

## How It Works

RAPHAEL uses a novel Mixture-of-Experts (MoE) approach in the cross-attention layers
of a diffusion U-Net. Each text concept activates different expert pathways,
enabling fine-grained alignment between text tokens and image regions.

Technical specifications:

- Architecture: U-Net with MoE cross-attention layers
- Expert routing: Top-k routing per token/region/timestep
- Text encoder: CLIP ViT-L + edge-of-concept attention
- Resolution: 1024x1024
- Latent channels: 4

Reference: Xue et al., "RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths", NeurIPS 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RAPHAELModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of RAPHAELModel with full customization support. |

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
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

