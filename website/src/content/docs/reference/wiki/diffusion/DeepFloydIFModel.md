---
title: "DeepFloydIFModel<T>"
description: "DeepFloyd IF model for cascaded text-to-image generation in pixel space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

DeepFloyd IF model for cascaded text-to-image generation in pixel space.

## For Beginners

DeepFloyd IF generates images by progressively upscaling:

How DeepFloyd IF works:

1. Stage I: Generates a 64x64 pixel image from text (using T5-XXL embeddings)
2. Stage II: Upscales 64x64 → 256x256 with text-guided super-resolution
3. Stage III: Upscales 256x256 → 1024x1024 (optional, non-diffusion upscaler)

Key characteristics:

- Pixel-space diffusion (no VAE/latent space for Stages I and II)
- Text encoder: Frozen T5-XXL (4.7B parameters, 4096-dim embeddings)
- Stage I: ~900M parameters, 64x64 output
- Stage II: ~450M parameters, 256x256 output
- Stage III: Non-diffusion upscaler to 1024x1024
- Exceptional text rendering in images

Advantages:

- Best-in-class text rendering (can write legible text)
- Exceptional prompt adherence from T5-XXL
- Pixel-space avoids VAE artifacts

Limitations:

- Very large memory requirement (T5-XXL alone is ~10GB)
- Slower than latent diffusion models
- Multi-stage pipeline adds complexity
- Restricted license (not fully open-source)

## How It Works

DeepFloyd IF is a three-stage cascaded diffusion model that operates in pixel space
(not latent space), developed by DeepFloyd (Stability AI). It uses a frozen T5-XXL
text encoder for exceptional text understanding and prompt adherence.

Technical specifications:

- Architecture: Cascaded pixel-space diffusion
- Stage I: U-Net, ~900M parameters, 64x64 output, 3 RGB channels
- Stage II: U-Net, ~450M parameters, 256x256 output, 6 channels (3 RGB + 3 low-res input)
- Text encoder: Frozen T5-XXL (4.7B params, 4096-dim, 256 max tokens)
- Noise schedule: Cosine beta schedule, 1000 training timesteps
- Prediction type: Epsilon prediction with dynamic thresholding

Reference: Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepFloydIFModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Boolean,Nullable<Int32>)` | Initializes a new instance of DeepFloydIFModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (4096 for T5-XXL). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `StageIINoisePredictor` | Gets the Stage II super-resolution noise predictor. |
| `UsesDynamicThresholding` | Gets whether dynamic thresholding is enabled. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `InitializeLayers(UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the Stage I (64x64), Stage II (256x256) U-Nets, and optional VAE, using custom layers from the user if provided or creating industry-standard layers from the Imagen/DeepFloyd IF paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for Stage I (64x64 base generation). |
| `DefaultWidth` | Default image width for Stage I (64x64 base generation). |
| `IF_CROSS_ATTENTION_DIM` | Cross-attention dimension matching T5-XXL output (4096). |
| `IF_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for DeepFloyd IF (7.0). |

