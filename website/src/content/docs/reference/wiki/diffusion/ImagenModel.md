---
title: "ImagenModel<T>"
description: "Imagen model for cascaded text-to-image generation with T5 text encoding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Imagen model for cascaded text-to-image generation with T5 text encoding.

## For Beginners

Imagen generates images by starting small and upscaling:

How Imagen works:

1. Text is encoded by a frozen T5-XXL language model (4096-dim embeddings)
2. A base diffusion model generates a 64x64 image from the text embeddings
3. A first super-resolution model upscales 64x64 → 256x256
4. A second super-resolution model upscales 256x256 → 1024x1024

Key characteristics:

- Cascaded pixel-space diffusion: 64→256→1024
- Text encoder: Frozen T5-XXL (4.7B parameters, 4096-dim)
- Base model: ~2B parameters, 64x64 output
- Super-res 1: ~600M parameters, 256x256 output
- Super-res 2: ~400M parameters, 1024x1024 output
- Uses Efficient U-Net architecture
- Dynamic thresholding for improved image quality

Key innovations:

- Demonstrated that text encoder quality is most important
- Introduced dynamic thresholding (better high guidance scales)
- Efficient U-Net: memory-efficient attention, shifted convolutions
- Noise conditioning augmentation for super-resolution stages

Limitations:

- Not open-source (proprietary to Google)
- Very large compute requirements
- Three separate models needed for full pipeline

## How It Works

Imagen is a cascaded text-to-image diffusion model developed by Google Brain.
It demonstrates that large frozen language models (T5-XXL) are highly effective for
text-to-image generation, and that scaling the text encoder matters more than scaling the image model.

Technical specifications:

- Architecture: Cascaded pixel-space diffusion with Efficient U-Net
- Base model: 64x64, ~2B parameters, 3 RGB channels
- Super-res 1: 64→256, ~600M parameters
- Super-res 2: 256→1024, ~400M parameters
- Text encoder: Frozen T5-XXL (4.7B params, 4096-dim, 256 max tokens)
- Noise schedule: Cosine schedule, 1000 training timesteps
- Prediction type: Epsilon prediction with dynamic thresholding
- Noise conditioning augmentation for super-res stages

Reference: Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", NeurIPS 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImagenModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Double,Nullable<Int32>)` | Initializes a new instance of ImagenModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (4096 for T5-XXL). |
| `DynamicThresholdPercentile` | Gets the dynamic thresholding percentile. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SuperResolution1` | Gets the super-resolution Stage 1 noise predictor (64→256). |
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
| `InitializeLayers(UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the Efficient U-Net base model, super-resolution model, and VAE, using custom layers from the user if provided or creating industry-standard layers from the Imagen paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for Imagen base model (64x64). |
| `DefaultWidth` | Default image width for Imagen base model (64x64). |
| `IMAGEN_CROSS_ATTENTION_DIM` | Cross-attention dimension matching T5-XXL output (4096). |
| `IMAGEN_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Imagen (7.5). |
| `IMAGEN_DYNAMIC_THRESHOLD_PERCENTILE` | Default dynamic thresholding percentile (99.5%). |

