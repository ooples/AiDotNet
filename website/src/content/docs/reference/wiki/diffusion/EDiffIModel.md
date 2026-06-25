---
title: "EDiffIModel<T>"
description: "eDiff-I model — ensemble of expert denoisers for text-to-image diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

eDiff-I model — ensemble of expert denoisers for text-to-image diffusion.

## For Beginners

eDiff-I uses teamwork among specialist networks:

How eDiff-I works:

1. Multiple specialized U-Nets are trained for different noise levels
2. High-noise expert: generates overall structure and composition
3. Mid-noise expert: adds medium-scale features and objects
4. Low-noise expert: refines fine details and textures
5. The right expert is used at each denoising step

Key characteristics:

- Ensemble of 3+ specialized denoiser networks
- Each expert trained on specific noise range
- Better quality than single-model approaches
- Paint-with-words: spatial control over generation

Use eDiff-I when you need:

- Maximum image quality
- Spatial control over concept placement
- Better structure-detail trade-off

## How It Works

eDiff-I uses an ensemble of specialized denoiser networks, each trained on a
specific range of noise levels. This allows each expert to focus on generating
specific aspects: structure (high noise) vs detail (low noise).

Technical specifications:

- Architecture: Ensemble of U-Nets with noise-level specialization
- Experts: 3 specialized denoisers (high/mid/low noise)
- Text encoder: T5-XXL + CLIP ensemble
- Resolution: 256x256 base, cascaded to 1024x1024
- Paint-with-words: region-based text-to-image control

Reference: Balaji et al., "eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers", 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EDiffIModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of EDiffIModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `NumExperts` | Gets the number of expert denoisers in the ensemble. |
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

