---
title: "SiDDiTModel<T>"
description: "SiD-DiT: Score Identity Distillation applied to Diffusion Transformer (DiT) architectures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

SiD-DiT: Score Identity Distillation applied to Diffusion Transformer (DiT) architectures.

## For Beginners

SiD-DiT is the transformer version of SiD. While SiD works with
U-Net models (like Stable Diffusion), SiD-DiT works with Diffusion Transformers (DiT)
— the architecture used by newer models. It enables single-step generation from these
powerful transformer-based generators.

## How It Works

Extends SiD to work with DiT-based architectures, enabling single-step generation from
transformer-based diffusion models. Uses the same score identity principle but adapted
for the DiT's class-conditional generation with adaptive layer norm (adaLN) conditioning.

Reference: Extended from "Score Identity Distillation" (Zhou et al., 2024) to DiT architectures

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SiDDiTModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,SiTPredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new SiD-DiT model. |

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

