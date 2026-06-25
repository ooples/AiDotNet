---
title: "EmuVideoModel<T>"
description: "Emu Video high-quality video generation with temporal consistency."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.AudioVisual`

Emu Video high-quality video generation with temporal consistency.

## For Beginners

Emu Video from Meta generates high-quality videos with strong temporal consistency, meaning objects maintain their appearance and motion stays smooth throughout the clip. It uses a factored approach separating image generation from temporal animation.

## How It Works

**References:**

- Paper: "Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning" (Meta, 2023)

Emu Video generates high-quality videos by factorizing the text-to-video problem into two stages:
first generating a conditioning image from text, then animating it. This factorization simplifies
the learning problem and produces videos with strong temporal consistency and natural motion.

Technical specifications:

- Architecture: Factorized T2V: Text-to-Image + Image-to-Video
- Latent channels: 16
- Default: 16 frames at 16 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EmuVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of EmuVideoModel with full customization support. |

