---
title: "SD3InpaintingModel<T>"
description: "Stable Diffusion 3 inpainting model using MMDiT architecture for mask-guided generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

Stable Diffusion 3 inpainting model using MMDiT architecture for mask-guided generation.

## For Beginners

This is Stable Diffusion 3's inpainting mode. SD3 uses a modern
transformer architecture (instead of U-Net) and three text encoders for better prompt
understanding. It produces high-quality inpainting results with excellent text following.

## How It Works

Adapts the SD3 MMDiT (Multimodal Diffusion Transformer) architecture for inpainting.
Uses 16-channel latent space with the MMDiT-X noise predictor for high-quality
mask-conditioned generation with improved text understanding via triple text encoders.

Technical specifications (SD3 Medium):

- Architecture: MMDiT-X (Multimodal Diffusion Transformer with cross-attention)
- Hidden size: 1536, 24 joint attention layers, 24 heads
- Text encoders: CLIP-L (768-dim) + CLIP-G (1280-dim) + T5-XXL (4096-dim)
- Latent space: 16 channels with rectified flow training
- Prediction type: v-prediction (velocity)
- Resolution: Up to 1024x1024 (aspect-ratio buckets)

Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024

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

