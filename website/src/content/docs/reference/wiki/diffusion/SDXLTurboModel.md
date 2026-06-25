---
title: "SDXLTurboModel<T>"
description: "SDXL Turbo model for real-time single-step high-resolution image generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SDXL Turbo model for real-time single-step high-resolution image generation.

## For Beginners

SDXL Turbo combines the image quality of SDXL (one of the best
open-source models) with near-instant generation. While regular SDXL needs 25-50 steps,
SDXL Turbo generates comparable images in just 1 step. No guidance needed (scale=0).

## How It Works

SDXL Turbo is the SDXL-based variant of Adversarial Diffusion Distillation (ADD),
generating 512x512 images in 1-4 steps with SDXL-quality aesthetics. Uses the full
SDXL U-Net with dual text encoder conditioning (CLIP ViT-L + OpenCLIP ViT-bigG).

Reference: Sauer et al., "Adversarial Diffusion Distillation", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SDXLTurboModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SDXL Turbo model. |

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
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

