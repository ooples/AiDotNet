---
title: "UniControlNetModel<T>"
description: "Uni-ControlNet model for simultaneous multi-condition control with condition-specific adapters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

Uni-ControlNet model for simultaneous multi-condition control with condition-specific adapters.

## For Beginners

Uni-ControlNet combines multiple spatial conditions simultaneously.

How Uni-ControlNet works:

1. Multiple condition images are provided (e.g., depth + edges + pose simultaneously)
2. Each condition is processed by its specific adapter encoder
3. Adapter features are mixed with learned weights for each condition
4. Combined control features are injected into the SD 1.5 U-Net
5. The U-Net generates an image guided by all conditions simultaneously

Key characteristics:

- Simultaneous multi-condition control (depth + edges + pose at same time)
- Condition-specific adapters with learned mixing weights
- Single forward pass for all conditions combined
- Compatible with SD 1.5 backbone
- No need to manually balance multiple ControlNets

When to use Uni-ControlNet:

- Multiple conditions applied simultaneously
- Complex spatial control combining depth, edges, and pose
- Single-model multi-condition pipeline
- When per-condition ControlNet loading is impractical

Limitations:

- SD 1.5 resolution (512x512)
- Adding new condition types requires retraining
- Condition interactions can produce unexpected results
- Not as flexible as separate ControlNets for per-condition tuning

## How It Works

Uni-ControlNet allows composing multiple visual conditions simultaneously in a single
forward pass. It uses condition-specific adapters that are mixed together with learned
weights, enabling complex multi-condition spatial control.

Architecture components:

- SD 1.5 U-Net backbone (320 base channels, [1,2,4,4], 768-dim CLIP)
- Condition-specific encoder adapters for each control type
- Learned condition mixing weights for multi-condition composition
- Standard SD 1.5 VAE for image encoding/decoding
- DDIM scheduler for efficient inference

Technical specifications:

- Architecture: Multi-adapter ControlNet with condition mixing
- Backbone: SD 1.5 (320 base channels, [1,2,4,4] multipliers)
- Cross-attention: 768-dim (CLIP ViT-L/14)
- Adapters: condition-specific encoders with shared backbone
- Default resolution: 512x512
- Scheduler: DDIM

Reference: Zhao et al., "Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models", NeurIPS 2023

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
| `GenerateFromText(String,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

