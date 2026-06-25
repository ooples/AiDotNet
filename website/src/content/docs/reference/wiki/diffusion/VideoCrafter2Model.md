---
title: "VideoCrafter2Model<T>"
description: "VideoCrafter 2 with improved quality and style fusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

VideoCrafter 2 with improved quality and style fusion.

## For Beginners

VideoCrafter 2 improves on its predecessor with better visual quality and the ability to blend artistic styles into generated videos. It combines quality improvements with style transfer capabilities for creative video generation.

## How It Works

**References:**

- Paper: "VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models" (Tencent, 2024)

VideoCrafter 2 improves video quality by overcoming data limitations through better data curation
and augmentation strategies. Uses a Video LDM architecture with factorized spatiotemporal attention
and CLIP text encoding. Achieves significant quality improvements over VideoCrafter 1 with
better temporal consistency and visual fidelity.

Technical specifications:

- Architecture: Video LDM + Factorized Spatiotemporal Attention + CLIP
- Latent channels: 4
- Default: 16 frames at 8 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoCrafter2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,VideoUNetPredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of VideoCrafter2Model with full customization support. |

