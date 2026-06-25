---
title: "Step1XEditModel<T>"
description: "Step1X-Edit model for one-step image editing using consistency distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Step1X-Edit model for one-step image editing using consistency distillation.

## For Beginners

Step1X-Edit is the fastest editing model — it makes changes
in a single step instead of the usual 20-50. This makes editing feel instant,
perfect for interactive tools where you want immediate feedback.

## How It Works

Step1X-Edit achieves single-step image editing through consistency distillation,
mapping directly from the source image and edit instruction to the edited result
in one forward pass. Built on a DiT backbone distilled from a multi-step teacher.

Technical specifications:

- Architecture: DiT backbone with consistency distillation
- Inference: Single-step (1 NFE) — no iterative denoising
- Latent space: 16 channels
- Distilled from multi-step rectified flow teacher
- Guidance: 1.0 (distilled models don't need CFG)

Reference: StepFun, "Step1X-Edit", 2025

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

