---
title: "HunyuanDiTModel<T>"
description: "Hunyuan-DiT model — bilingual (Chinese-English) DiT text-to-image model by Tencent."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.TextToImage`

Hunyuan-DiT model — bilingual (Chinese-English) DiT text-to-image model by Tencent.

## For Beginners

Hunyuan-DiT generates images from Chinese or English prompts:

Key characteristics:

- Bilingual: understands both Chinese and English prompts natively
- DiT backbone: transformer-based denoiser (1.5B parameters)
- Dual text encoders: CLIP-L/14 + mT5-XL for multilingual support
- Multi-resolution training with aspect ratio bucketing
- Human preference alignment via RLHF-like training

How Hunyuan-DiT works:

1. Text goes through CLIP (visual) + mT5 (multilingual) encoders
2. DiT transformer denoises with cross-attention to both embeddings
3. Multi-resolution support through positional embedding interpolation
4. VAE decoder produces final image

Use Hunyuan-DiT when you need:

- Chinese text-to-image generation
- Bilingual applications
- Open-source multilingual alternative

## How It Works

Hunyuan-DiT is Tencent's bilingual text-to-image model that uses a DiT transformer
backbone with dual text encoders (CLIP + multilingual T5) for both Chinese and
English prompt understanding.

Technical specifications:

- Architecture: DiT-XL with dual text conditioning
- Parameters: ~1.5B (DiT backbone)
- Text encoders: CLIP ViT-L/14 (768-dim) + mT5-XL (2048-dim)
- Native resolution: 1024x1024
- Latent space: 4 channels, 8x downsampling
- Guidance scale: 6.0 recommended

Reference: Li et al., "Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer
with Fine-Grained Chinese Understanding", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HunyuanDiTModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new instance of HunyuanDiTModel with full customization support. |

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

