---
title: "Flux1Model<T>"
description: "FLUX.1 model for high-quality text-to-image generation by Black Forest Labs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

FLUX.1 model for high-quality text-to-image generation by Black Forest Labs.

## For Beginners

FLUX.1 represents the cutting edge of open text-to-image models:

How FLUX.1 works:

1. Text is encoded by CLIP ViT-L/14 and T5-XXL encoders
2. A hybrid MMDiT with 19 joint blocks + 38 single blocks processes tokens
3. Rectified flow matching enables efficient generation
4. A 16-channel VAE decodes latents to high-resolution images

Model variants:

- FLUX.1 [pro]: Best quality, API-only, not open-source
- FLUX.1 [dev]: Open-weight, guidance-distilled, non-commercial license
- FLUX.1 [schnell]: Fast 1-4 step generation, Apache 2.0 license

Key characteristics:

- 12B parameters in the transformer
- Hybrid architecture: 19 double-stream + 38 single-stream blocks
- Hidden size: 3072, 24 attention heads
- Dual text encoders: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
- 16 latent channels with new VAE
- Rotary Position Embeddings (RoPE)
- Rectified flow with linear noise schedule

Advantages:

- State-of-the-art image quality
- Excellent text rendering
- Superior prompt adherence
- schnell variant: very fast (1-4 steps)
- dev variant: open weights for research

Limitations:

- Very high VRAM requirements (~24GB for dev)
- pro variant is API-only
- Newer ecosystem (fewer community tools)

## How It Works

FLUX.1 is a state-of-the-art text-to-image model developed by Black Forest Labs
(founded by Stability AI alumni). It uses a hybrid MMDiT architecture with both
double-stream (joint attention) and single-stream transformer blocks, plus rectified
flow matching for training.

Technical specifications:

- Architecture: Hybrid MMDiT (double-stream + single-stream blocks)
- Transformer: 12B params, hidden 3072, 19 joint + 38 single layers, 24 heads
- Text encoder 1: CLIP ViT-L/14 (768-dim, pooled embeddings)
- Text encoder 2: T5-XXL (4096-dim, sequence embeddings)
- Context dimension: 4096 (T5 embeddings)
- Patch size: 2 (in latent space)
- VAE: 16 latent channels
- Training: Rectified flow matching
- dev: 50-step guidance-distilled
- schnell: 1-4 step distilled
- Resolution: Up to 2048x2048 (aspect-ratio aware)

Reference: Black Forest Labs, "FLUX.1 Technical Report", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Flux1Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,MMDiTNoisePredictor<>,StandardVAE<>,IConditioningModule<>,FluxVariant,Nullable<Int32>)` | Initializes a new instance of Flux1Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsGuidanceFree` | Gets whether this variant supports guidance-free generation. |
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
| `DefaultHeight` | Default image height for FLUX.1 (1024x1024). |
| `DefaultWidth` | Default image width for FLUX.1 (1024x1024). |

