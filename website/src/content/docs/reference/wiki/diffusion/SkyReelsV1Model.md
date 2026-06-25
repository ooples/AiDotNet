---
title: "SkyReelsV1Model<T>"
description: "SkyReels V1 human-centric video generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

SkyReels V1 human-centric video generation model.

## For Beginners

SkyReels V1 is a video generation model trained on 10 million film clips, specializing in realistic human-centric content. It excels at generating natural human motion, facial expressions, and interactions in cinematic style.

## How It Works

**References:**

- Paper: "SkyReels V1: Human-Centric Video Foundation Model" (Kunlun Tech, 2025)

SkyReels V1 is a specialized human-centric video model built by fine-tuning HunyuanVideo on
approximately 10 million high-quality film and television clips. It focuses on realistic human
portrayals with cinematic quality, particularly excelling at facial expressions, body movements,
and human-environment interaction.

Technical specifications:

- Architecture: DiT (HunyuanVideo-based) + Human-Centric Fine-tuning
- Latent channels: 16
- Default: 49 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SkyReelsV1Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of SkyReelsV1Model with full customization support. |

