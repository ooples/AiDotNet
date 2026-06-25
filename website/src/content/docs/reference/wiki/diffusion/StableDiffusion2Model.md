---
title: "StableDiffusion2Model<T>"
description: "Stable Diffusion 2.0/2.1 model for text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Stable Diffusion 2.0/2.1 model for text-to-image generation.

## For Beginners

SD 2.0/2.1 is an upgraded version of SD 1.5 with key differences:

How SD 2.x works:

1. Your text prompt is encoded by OpenCLIP ViT-H/14 (1024-dim embeddings)
2. These embeddings guide a U-Net (865M parameters) that denoises in latent space
3. A VAE decodes the denoised latent into a 768x768 image (SD 2.0) or 512/768 (SD 2.1)

Key differences from SD 1.5:

- Text encoder: OpenCLIP ViT-H/14 (1024-dim) vs CLIP ViT-L/14 (768-dim)
- Prediction type: v-prediction vs epsilon-prediction
- Native resolution: 768x768 (SD 2.0) or 512x512/768x768 (SD 2.1)
- Removed NSFW content from training data
- Better at generating text in images

When to use SD 2.x:

- Need v-prediction for certain workflows
- Want OpenCLIP text encoder (different strengths than CLIP)
- Need 768x768 native resolution

Limitations:

- Smaller community ecosystem than SD 1.5
- Some users prefer SD 1.5's CLIP encoder for prompt adherence
- Not as widely supported by third-party tools

## How It Works

Stable Diffusion 2.x is the second generation of Stability AI's text-to-image model.
It uses OpenCLIP ViT-H/14 instead of the original CLIP ViT-L/14 text encoder,
and introduces v-prediction as the default prediction type.

Technical specifications:

- Architecture: U-Net with cross-attention and time embedding
- Text encoder: OpenCLIP ViT-H/14 (632M parameters, 1024-dim, 77 max tokens)
- U-Net: 865M parameters, base channels 320, [1, 2, 4, 4] multipliers
- Cross-attention dimension: 1024 (matches OpenCLIP output)
- VAE: KL-regularized autoencoder, 4 latent channels, scale factor 0.18215
- Noise schedule: Scaled linear beta schedule, 1000 training timesteps
- Prediction type: v-prediction (velocity prediction)

Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableDiffusion2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Boolean,Nullable<Int32>)` | Initializes a new instance of StableDiffusion2Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (1024 for SD 2.x, matching OpenCLIP ViT-H/14). |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `UsesVPrediction` | Gets whether this model uses v-prediction (velocity prediction). |
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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net and VAE layers, using custom layers from the user if provided or creating industry-standard layers from the SD 2.x paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for SD 2.x generation. |
| `DefaultWidth` | Default image width for SD 2.x generation. |
| `SD2_CROSS_ATTENTION_DIM` | Cross-attention dimension matching OpenCLIP ViT-H/14 output (1024). |
| `SD2_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for SD 2.x (7.5). |

