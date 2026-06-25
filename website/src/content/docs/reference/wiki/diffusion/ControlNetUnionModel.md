---
title: "ControlNetUnionModel<T>"
description: "ControlNet Union model for unified multi-condition image generation with a single ControlNet."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet Union model for unified multi-condition image generation with a single ControlNet.

## For Beginners

ControlNet Union is an all-in-one control model that replaces 8+ separate ControlNets.

How ControlNet Union works:

1. A condition image (depth map, canny edges, pose, etc.) is provided as input
2. A task routing token tells the model which condition type is being used
3. The unified ControlNet encoder processes the condition with task-specific pathways
4. ControlNet features are injected into the SDXL U-Net via skip connections
5. The U-Net generates the final image guided by both text prompt and spatial condition

Supported conditions:

- Canny edges, depth maps, normal maps
- OpenPose skeleton, segmentation maps
- Scribbles, line art
- Low-quality image tile upscaling

When to use ControlNet Union:

- Multiple control types without loading multiple models
- Memory-efficient multi-condition pipeline
- Mixed-condition generation in a single forward pass
- SDXL-resolution (1024x1024) controlled generation

Limitations:

- SDXL-only (not compatible with SD 1.5)
- Slightly lower per-condition quality than specialized ControlNets
- Task routing adds small overhead
- Limited to supported condition types

## How It Works

ControlNet Union combines multiple control conditions (depth, canny, pose, normal, scribble,
lineart, segmentation, tile) into a single ControlNet model. Instead of loading separate
ControlNets for each condition type, one unified model handles all conditions via
task-specific routing tokens.

Architecture components:

- SDXL-compatible U-Net backbone (320 base channels, [1,2,4], 2048-dim dual encoder)
- Unified ControlNet encoder with task routing tokens for 8+ condition types
- SDXL VAE with 0.13025 latent scale factor
- Task-specific embedding layer for condition type selection
- Euler discrete scheduler for efficient inference

Technical specifications:

- Architecture: ControlNet with task routing tokens
- Base U-Net: SDXL (320 base channels, [1,2,4] multipliers)
- Cross-attention: 2048-dim (dual text encoder)
- Conditions: 8+ types via task embedding
- Default resolution: 1024x1024
- Scheduler: Euler discrete
- Replaces: 8+ individual ControlNets

Reference: Xinsong Zhang, "ControlNet++/Union", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `ControlNet` | Gets the unified control network. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

