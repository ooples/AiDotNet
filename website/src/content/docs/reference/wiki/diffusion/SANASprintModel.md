---
title: "SANASprintModel<T>"
description: "SANA Sprint model for ultra-fast 1-step generation from the SANA architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SANA Sprint model for ultra-fast 1-step generation from the SANA architecture.

## For Beginners

SANA is already fast due to its efficient architecture.
SANA Sprint makes it even faster — single-step generation at 1024x1024 resolution.
This makes it one of the fastest high-resolution generators available, suitable for
real-time applications like interactive image editing.

## How It Works

SANA Sprint is the distilled version of SANA that generates 1024x1024 images in a
single step. Uses the efficient linear attention DiT backbone from SANA combined
with hybrid distillation (consistency + adversarial) for real-time generation.

Reference: NVIDIA, "SANA Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation", 2025

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SANASprintModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,SiTPredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SANA Sprint model. |

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

