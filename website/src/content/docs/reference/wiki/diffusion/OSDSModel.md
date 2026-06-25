---
title: "OSDSModel<T>"
description: "One-Step Diffusion via Shortcut (OSDS) model for single-step high-quality generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

One-Step Diffusion via Shortcut (OSDS) model for single-step high-quality generation.

## For Beginners

Most diffusion models need many steps to turn noise into an image.
OSDS learns a "shortcut" — a direct path from noise to image in just one step. Think of
it like learning both the scenic route (multi-step) and a highway (one-step) to the same
destination. You can choose speed or quality depending on your needs.

## How It Works

OSDS learns a shortcut function that directly maps noise to clean data in a single
forward pass. Unlike distillation approaches, it trains a dedicated shortcut path
that bypasses the iterative denoising process entirely. The shortcut is learned
alongside the standard diffusion trajectory, enabling both one-step and multi-step
generation from the same model.

Reference: Frans et al., "One Step Diffusion via Shortcut Models", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OSDSModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new OSDS model with optional configuration. |

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

