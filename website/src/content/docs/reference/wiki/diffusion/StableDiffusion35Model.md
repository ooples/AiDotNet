---
title: "StableDiffusion35Model<T>"
description: "Stable Diffusion 3.5 model with improved MMDiT-X architecture by Stability AI."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Stable Diffusion 3.5 model with improved MMDiT-X architecture by Stability AI.

## For Beginners

Stable Diffusion 3.5 is Stability AI's latest open model.

How SD 3.5 works:

1. Text is encoded by three encoders: CLIP ViT-L/14, OpenCLIP ViT-bigG, and T5-XXL
2. An improved MMDiT-X transformer with QK-normalization processes text and image tokens
3. Rectified flow matching enables efficient 28-40 step generation
4. A 16-channel VAE decodes latents to high-resolution images

Model variants:

- SD 3.5 Medium: 2.5B parameters, faster, good quality-speed tradeoff
- SD 3.5 Large: 8B parameters, highest quality, more VRAM required

Key characteristics:

- MMDiT-X architecture with QK-normalization for stable training
- Triple text encoders for superior prompt understanding
- 16 latent channels with improved VAE
- Rectified flow matching (not DDPM/DDIM)
- Medium: 2.5B params, 1536 hidden, 24 layers
- Large: 8B params, 4096 hidden, 38 layers

Advantages:

- Open-weight (Stability AI Community License)
- State-of-the-art quality among open models
- Better prompt adherence than SDXL
- QK-norm prevents training instabilities

Limitations:

- Large variant requires significant VRAM (~24GB)
- Newer ecosystem than SD 1.5/SDXL
- Commercial use requires separate license agreement

## How It Works

SD 3.5 builds on SD3's MMDiT architecture with improved text-image alignment,
better detail generation, and QK-normalization for training stability. Uses
rectified flow matching training and triple text encoders for comprehensive
prompt understanding. Available in Medium (2.5B) and Large (8B) variants.

Technical specifications:

- Architecture: MMDiT-X with QK-normalization
- Medium: 2.5B params, hidden 1536, 24 layers, 24 heads
- Large: 8B params, hidden 4096, 38 layers, 64 heads
- Text encoder 1: CLIP ViT-L/14 (768-dim, pooled)
- Text encoder 2: OpenCLIP ViT-bigG (1280-dim, pooled)
- Text encoder 3: T5-XXL (4096-dim, sequence)
- Patch size: 2 (in latent space)
- VAE: 16 latent channels, 8x spatial compression
- Training: Rectified flow matching
- Resolution: 1024x1024 default, up to 2048x2048

Reference: Esser et al., "Scaling Rectified Flow Transformers for
High-Resolution Image Synthesis", ICML 2024 (SD3 base);
Stability AI, "Stable Diffusion 3.5 Release Notes", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StableDiffusion35Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MMDiTXNoisePredictor<>,StandardVAE<>,IConditioningModule<>,MMDiTXVariant,Nullable<Int32>)` | Initializes a new instance of StableDiffusion35Model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |
| `Variant` | Gets the model variant (Medium or Large). |

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
| `DefaultHeight` | Default image height for SD 3.5 (1024x1024). |
| `DefaultWidth` | Default image width for SD 3.5 (1024x1024). |

