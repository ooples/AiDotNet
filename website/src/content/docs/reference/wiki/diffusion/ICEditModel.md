---
title: "ICEditModel<T>"
description: "ICEdit model for in-context learning based image editing without per-task fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

ICEdit model for in-context learning based image editing without per-task fine-tuning.

## For Beginners

ICEdit learns editing by example. You show it a "before" and
"after" pair (e.g., a photo and its black-and-white version), and it applies the
same transformation to new images. No special training needed for each edit type.

## How It Works

ICEdit leverages in-context learning to understand editing instructions from examples.
Given an instruction and optional before/after examples, it learns the edit pattern
and applies it to new images without requiring task-specific training. Built on the
FLUX architecture for high-quality generation.

Technical specifications:

- Base model: FLUX-based architecture (rectified flow transformer)
- Editing approach: In-context learning with example pairs
- No per-task fine-tuning required — learns from demonstrations
- Hidden size: 3072, 16 latent channels
- Text encoders: CLIP ViT-L/14 + T5-XXL (FLUX dual encoder)

Reference: Zhang et al., "ICEdit: In-Context Image Editing", 2025

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

