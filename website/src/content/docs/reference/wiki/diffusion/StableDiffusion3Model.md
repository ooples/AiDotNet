---
title: "StableDiffusion3Model<T>"
description: "Stable Diffusion 3 / SD 3.5 model for text-to-image generation using rectified flow and MMDiT."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Stable Diffusion 3 / SD 3.5 model for text-to-image generation using rectified flow and MMDiT.

## For Beginners

SD3 is the successor to SDXL with major architectural improvements:

How SD3 works:

1. Text is encoded by THREE encoders: CLIP ViT-L/14, OpenCLIP ViT-bigG/14, and T5-XXL
2. An MMDiT (Multi-Modal Diffusion Transformer) processes image and text tokens jointly
3. Uses rectified flow instead of DDPM noise scheduling
4. A new 16-channel VAE decodes latents to 1024x1024 images

Key characteristics:

- Triple text encoders (CLIP L + OpenCLIP G + T5-XXL)
- MMDiT: Joint attention between text and image tokens
- 16 latent channels (vs 4 in SD 1.5/SDXL)
- Rectified flow (linear noise schedule, v-prediction)
- SD3 Medium: 2B MMDiT parameters, 24 layers
- SD3.5 Large: 8B MMDiT parameters, 38 layers
- SD3.5 Large Turbo: 8B distilled for 4-step generation

Advantages:

- Superior text rendering in generated images
- Better prompt adherence than SDXL
- Higher quality details and compositions
- Scalable MMDiT architecture

Limitations:

- Higher compute requirements than SDXL
- Fewer community fine-tunes (newer ecosystem)
- T5-XXL encoder increases VRAM usage significantly

## How It Works

Stable Diffusion 3 (SD3) is a next-generation text-to-image model by Stability AI that
replaces the U-Net with a Multi-Modal Diffusion Transformer (MMDiT) and uses rectified
flow matching instead of traditional DDPM-style noise schedules.

Technical specifications:

- Architecture: MMDiT (Multi-Modal Diffusion Transformer) + 16-channel VAE
- Text encoder 1: CLIP ViT-L/14 (768-dim)
- Text encoder 2: OpenCLIP ViT-bigG/14 (1280-dim)
- Text encoder 3: T5-XXL (4096-dim)
- Combined pooled embedding: 2048-dim (768 + 1280)
- Context dimension: 4096 (T5 embeddings for cross-attention)
- SD3 Medium: 2B params, hidden 1536, 24 layers, 24 heads
- SD3.5 Large: 8B params, hidden 2432, 38 layers, 38 heads
- VAE: 16 latent channels, scale factor 1.5305, shift 0.0609
- Training: Rectified flow matching with linear schedule
- Resolution: 1024x1024 (native)

Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableDiffusion3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MMDiTNoisePredictor<>,StandardVAE<>,IConditioningModule<>,SD3Variant,Nullable<Int32>)` | Initializes a new instance of StableDiffusion3Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |
| `Variant` | Gets the model variant. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `ImageToImage(Tensor<>,String,String,Double,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for SD3 (1024x1024). |
| `DefaultWidth` | Default image width for SD3 (1024x1024). |

