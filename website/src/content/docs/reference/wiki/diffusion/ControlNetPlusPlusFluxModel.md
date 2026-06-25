---
title: "ControlNetPlusPlusFluxModel<T>"
description: "ControlNet++ adapted for FLUX architecture with reward-guided training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet++ adapted for FLUX architecture with reward-guided training.

## For Beginners

This is the most advanced version of ControlNet that works
with FLUX models. It combines the improved training (ControlNet++) with FLUX's
powerful architecture for the best possible control over generated images.

## How It Works

Combines ControlNet++ reward-guided training with FLUX's flow-matching architecture.
Uses 16-channel latent space with double-stream transformer blocks for improved
control signal adherence on FLUX-based models.

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

