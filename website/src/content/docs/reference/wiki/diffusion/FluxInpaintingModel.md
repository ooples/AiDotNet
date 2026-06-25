---
title: "FluxInpaintingModel<T>"
description: "FLUX Fill model for mask-guided inpainting using rectified flow transformers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

FLUX Fill model for mask-guided inpainting using rectified flow transformers.

## For Beginners

This is inpainting built on the FLUX architecture, which is
known for excellent image quality. It fills in masked regions using FLUX's advanced
transformer-based generation, producing results with 16-channel latent precision
and superior text understanding from dual encoders.

## How It Works

Adapts the FLUX rectified flow architecture for inpainting by conditioning on both
the masked image latent and the binary mask. Uses the FLUX hybrid MMDiT with
19 double-stream + 38 single-stream transformer blocks for high-quality 16-channel
latent inpainting with dual text encoder conditioning (CLIP + T5).

Technical specifications:

- Architecture: FLUX hybrid MMDiT (19 joint + 38 single blocks)
- Hidden size: 3072, 24 attention heads
- Text encoders: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
- Latent space: 16 channels, patch size 2
- Training: Rectified flow matching
- Resolution: Up to 2048x2048 (aspect-ratio aware)

Reference: Black Forest Labs, "FLUX.1 Fill", 2024

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

