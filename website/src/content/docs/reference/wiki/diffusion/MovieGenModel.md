---
title: "MovieGenModel<T>"
description: "MovieGen 30B foundation model for video, audio, and editing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

MovieGen 30B foundation model for video, audio, and editing.

## For Beginners

Movie Gen from Meta is a massive 30B parameter model that generates 16-second video clips at 16 FPS with synchronized audio. It is one of the largest video generation models, designed for high-fidelity movie-quality content.

## How It Works

**References:**

- Paper: "Movie Gen: A Cast of Media Foundation Models" (Meta, 2024)

MovieGen is Meta's 30B parameter foundation model generating 1080p HD videos at 16 FPS with
synchronized audio. Trained with a maximum context of 73K video tokens for 16-second generation.
Sets state-of-the-art across text-to-video synthesis, video personalization, video editing,
video-to-audio generation, and text-to-audio generation.

Technical specifications:

- Architecture: 30B DiT + Audio Generation + Video Editing
- Latent channels: 16
- Default: 256 frames at 16 FPS
- Supports I2V: Yes | T2V: Yes | V2V: Yes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MovieGenModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of MovieGenModel with full customization support. |

