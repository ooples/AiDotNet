---
title: "StableCascadeModel<T>"
description: "Stable Cascade (Würstchen v3) model for high-resolution text-to-image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Stable Cascade (Würstchen v3) model for high-resolution text-to-image generation.

## For Beginners

Stable Cascade generates images in three stages, like a relay race:

How Stable Cascade works:

1. Stage C (Prior): Generates a tiny 24x24 latent from your text prompt
2. Stage B (Decoder): Expands the 24x24 latent to a 256x256 latent
3. Stage A (VQGAN): Decodes the 256x256 latent to a full 1024x1024 image

Key characteristics:

- Three-stage cascade: Stage C (prior) → Stage B (decoder) → Stage A (VQGAN)
- Extreme compression: 42:1 spatial ratio (vs 8:1 for SD 1.5)
- Stage C operates in a very small 24×24 latent space (24 channels)
- Stage B: ~700M parameters, denoising diffusion
- Stage A: VQGAN decoder (frozen, non-diffusion)
- Text encoder: CLIP ViT-G/14 (1280-dim embeddings)
- Native resolution: 1024x1024

Advantages over Stable Diffusion:

- Much faster training and inference due to extreme compression
- Lower VRAM requirements for training
- Native 1024x1024 without quality degradation
- Better text-image alignment

Limitations:

- Smaller community ecosystem than SD 1.5/SDXL
- Three-stage pipeline is more complex to customize
- Fewer fine-tuned models available

## How It Works

Stable Cascade is a three-stage cascaded latent diffusion model developed by Stability AI,
based on the Würstchen architecture. It achieves extreme compression (42:1 spatial ratio)
allowing fast, high-quality 1024x1024 generation with lower compute requirements.

Technical specifications:

- Architecture: Three-stage cascaded latent diffusion
- Stage C: Würstchen prior, 1B parameters, 24-channel 24×24 latent
- Stage B: Würstchen decoder, ~700M parameters, 4-channel 256×256 latent
- Stage A: VQGAN (EfficientNet-based), frozen during training
- Text encoder: CLIP ViT-G/14 (1280-dim, 77 max tokens)
- Compression ratio: 42:1 spatial (1024→24)
- Noise schedule: Linear beta, 1000 training timesteps
- Prediction type: Epsilon prediction

Reference: Pernias et al., "Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableCascadeModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of StableCascadeModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `CrossAttentionDim` | Gets the cross-attention dimension (1280 for CLIP ViT-G/14). |
| `DecoderNoisePredictor` | Gets the Stage B decoder noise predictor. |
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
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `InitializeLayers(UNetNoisePredictor<>,UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the Stage C prior, Stage B decoder, and Stage A VQGAN layers, using custom layers from the user if provided or creating industry-standard layers from the Würstchen paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CASCADE_CROSS_ATTENTION_DIM` | Cross-attention dimension matching CLIP ViT-G/14 output (1280). |
| `CASCADE_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for Stable Cascade (4.0, lower than SD due to better alignment). |
| `DefaultHeight` | Default image height for Stable Cascade generation. |
| `DefaultWidth` | Default image width for Stable Cascade generation. |

