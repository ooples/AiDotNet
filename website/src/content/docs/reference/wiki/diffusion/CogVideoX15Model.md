---
title: "CogVideoX15Model<T>"
description: "CogVideoX 1.5 model with 10-second any-resolution video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

CogVideoX 1.5 model with 10-second any-resolution video generation.

## For Beginners

CogVideoX 1.5 creates 10-second videos at any resolution from text prompts. It compresses video efficiently using a 3D causal VAE (reducing memory 128x) and uses specialized expert transformer blocks that understand both spatial layout and temporal motion.

## How It Works

**References:**

- Paper: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (Zhipu AI/THUDM, 2024)

CogVideoX 1.5 extends the original CogVideoX to 10-second generation with any-resolution support.
Uses 3D causal VAE with 4x temporal and 8x spatial compression, expert transformer blocks with
adaptive layer normalization, and T5 text encoding. The 5B parameter model supports both
text-to-video and image-to-video generation at up to 720p.

Technical specifications:

- Architecture: 3D Causal VAE + Expert Transformer + T5
- Latent channels: 16
- Default: 80 frames at 8 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CogVideoX15Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of CogVideoX15Model with full customization support. |

