---
title: "Flux2Model<T>"
description: "FLUX.2 model for next-generation text-to-image generation by Black Forest Labs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

FLUX.2 model for next-generation text-to-image generation by Black Forest Labs.

## For Beginners

FLUX.2 is the improved version of FLUX.1, generating even better
images with sharper details and more accurate text rendering.

How FLUX.2 works:

1. Text is encoded by CLIP ViT-L/14 and T5-XXL encoders (dual encoder design)
2. A hybrid MMDiT with 19 joint blocks + 38 single blocks processes tokens
3. Improved rectified flow matching enables efficient 28-step generation
4. A 16-channel VAE decodes latents to high-resolution images

Model variants:

- FLUX.2 [pro]: Best quality, API-only, not open-source
- FLUX.2 [dev]: Open-weight, guidance-distilled, non-commercial license
- FLUX.2 [schnell]: Fast 1-4 step generation, Apache 2.0 license

Key improvements over FLUX.1:

- Better text rendering and prompt adherence
- Higher image quality at fewer inference steps (28 vs 50)
- Improved color accuracy and composition
- Enhanced fine detail generation

Technical characteristics:

- 12B parameters in the transformer
- Hybrid architecture: 19 double-stream + 38 single-stream blocks
- Hidden size: 3072, 24 attention heads
- Dual text encoders: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
- 16 latent channels with improved VAE
- Rotary Position Embeddings (RoPE)
- Improved rectified flow with optimized noise schedule

Limitations:

- Very high VRAM requirements (~24GB for dev)
- pro variant is API-only
- Newer model with evolving community support

## How It Works

FLUX.2 is the successor to FLUX.1, featuring improved image quality, better text rendering,
enhanced prompt adherence, and faster inference. It maintains the hybrid MMDiT architecture
with improvements to the attention mechanism, flow matching schedule, and training procedure.

Reference: Black Forest Labs, "FLUX.2 Technical Report", 2025

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Flux2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,FluxDoubleStreamPredictor<>,StandardVAE<>,IConditioningModule<>,FluxVariant,Nullable<Int32>)` | Initializes a new instance of Flux2Model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `IsGuidanceFree` | Gets whether this variant supports guidance-free generation. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |
| `Variant` | Gets the model variant (Dev, Schnell, or Pro). |

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
| `DefaultHeight` | Default image height for FLUX.2 (1024x1024). |
| `DefaultWidth` | Default image width for FLUX.2 (1024x1024). |

