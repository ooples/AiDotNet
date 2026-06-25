---
title: "LatentConsistencyModel<T>"
description: "Latent Consistency Model (LCM) for fast few-step image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Latent Consistency Model (LCM) for fast few-step image generation.

## For Beginners

LCM makes Stable Diffusion much faster:

How LCM works:

1. Starts with a pre-trained SD model (1.5, 2.1, or SDXL)
2. Uses consistency distillation to learn a shortcut through the denoising process
3. The distilled model can skip most denoising steps (2-8 instead of 20-50)
4. LCM-LoRA allows applying this speedup to ANY fine-tuned model

Key characteristics:

- 2-8 step generation (vs 20-50 for standard SD)
- Based on consistency distillation (not adversarial training)
- Lower guidance scales needed (1.0-2.0 vs 7.5)
- LCM-LoRA variant: lightweight adapter compatible with any SD fine-tune
- Compatible with SD 1.5, SD 2.1, and SDXL base models

Advantages:

- 5-10x faster than standard SD
- LCM-LoRA is composable with other LoRAs
- Maintains good quality at 4 steps
- Open-source and widely available

Limitations:

- Slightly lower diversity than full-step models
- Very low step counts (1-2) may show artifacts
- Lower guidance scales may reduce prompt adherence

## How It Works

Latent Consistency Models (LCM) distill pre-trained latent diffusion models into
fast inference models that can generate high-quality images in 2-8 steps.
LCM uses consistency distillation in latent space to learn a one-step mapping.

Technical specifications:

- Architecture: Consistency-distilled latent diffusion model
- Base: SD 1.5/2.1/SDXL U-Net with consistency training
- Distillation: Latent Consistency Distillation (LCD)
- Optimal steps: 4 (good quality/speed tradeoff)
- Guidance scale: 1.0-2.0 (lower than standard SD)
- Scheduler: LCM scheduler with skipping timesteps
- LCM-LoRA: ~67M adapter parameters for any SD fine-tune

Reference: Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LatentConsistencyModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,String,Nullable<Int32>)` | Initializes a new instance of LatentConsistencyModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseModel` | Gets the base model identifier ("SD1.5", "SD2.1", or "SDXL"). |
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` | Counts the flat-API parameter surface (predictor + VAE). |
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
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the consistency-distilled U-Net and VAE layers, using custom layers from the user if provided or creating industry-standard layers from the LCM paper. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultHeight` | Default image height for LCM (matches SD 1.5). |
| `DefaultWidth` | Default image width for LCM (matches SD 1.5). |
| `LCM_CROSS_ATTENTION_DIM` | Cross-attention dimension (768 for SD 1.5 base, 1024 for SD 2.1 base). |
| `LCM_DEFAULT_GUIDANCE_SCALE` | Default guidance scale for LCM (1.5, much lower than standard SD). |

