---
title: "MinimaxVideoModel<T>"
description: "MiniMax Hailuo video model with strong image-to-video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

MiniMax Hailuo video model with strong image-to-video generation.

## For Beginners

Minimax Video (Hailuo) excels at image-to-video generation, animating still images into natural-looking video clips. It produces smooth motion from a single reference image while maintaining the original image style.

## How It Works

**References:**

- Reference: MiniMax Hailuo I2V-01-Live (2024)

MiniMax Hailuo produces some of the most realistic AI-generated videos with impressive camera
control for cinematic scenes. Enhanced motion rendering provides smoother, more natural character
movements with near-photorealistic results in lighting, shadows, and color tones.

Technical specifications:

- Architecture: DiT + Enhanced Motion Rendering
- Latent channels: 16
- Default: 120 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MinimaxVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of MinimaxVideoModel with full customization support. |

