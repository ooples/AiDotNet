---
title: "UltraEditModel<T>"
description: "UltraEdit model for fine-grained instruction-based image editing with region awareness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ImageEditing`

UltraEdit model for fine-grained instruction-based image editing with region awareness.

## For Beginners

UltraEdit is a more precise version of text-based image editing.
When you say "change the sky color," it only modifies the sky and leaves everything
else untouched, unlike simpler models that might accidentally change other parts too.

## How It Works

UltraEdit improves upon InstructPix2Pix with region-aware editing that localizes changes
to relevant areas. Uses automatically generated editing masks from LLM-based instruction
parsing to prevent unwanted modifications to uninstructed regions. Built on SD1.5 backbone.

Technical specifications:

- Base model: Stable Diffusion 1.5 (InstructPix2Pix variant)
- Text encoder: CLIP ViT-L/14 (768-dim)
- Input channels: 8 (4 latent + 4 source image latent)
- Region awareness: LLM-generated editing masks from instructions
- Training data: Large-scale instruction-image pairs with automatic masks

Reference: Zhao et al., "UltraEdit: Instruction-based Fine-Grained Image Editing at Scale", 2024

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

