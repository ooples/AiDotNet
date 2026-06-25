---
title: "OmniGen2Model<T>"
description: "OmniGen-2 unified model for multi-task image generation and editing with a single architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

OmniGen-2 unified model for multi-task image generation and editing with a single architecture.

## For Beginners

OmniGen-2 is a "do everything" image model. Whether you want
to generate new images, edit existing ones, or copy the style of a reference image,
it handles all these tasks with one model instead of needing separate specialized models.
It understands instructions through a built-in language model (Phi-3).

## How It Works

OmniGen-2 is a unified model that handles text-to-image generation, image editing,
subject-driven generation, and visual conditional generation within a single
transformer-based architecture. It uses interleaved text-image input sequences
with a Phi-3 language model backbone for instruction understanding.

Technical specifications:

- Architecture: Transformer with Phi-3 Medium backbone
- Hidden size: 3072, 32 layers, 32 attention heads
- Text understanding: Phi-3 Medium (3.8B params) for instruction parsing
- Latent space: 16 channels (shared SDXL/FLUX VAE)
- Training: Rectified flow matching with interleaved text-image sequences
- Tasks: T2I, editing, subject-driven, visual conditioning (unified)

Reference: Xiao et al., "OmniGen: Unified Image Generation", 2024

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

