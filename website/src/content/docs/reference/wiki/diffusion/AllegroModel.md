---
title: "AllegroModel<T>"
description: "Allegro efficient DiT-based video generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Allegro efficient DiT-based video generation model.

## For Beginners

Allegro is an open-source AI video generator that achieves commercial-level quality with only 3 billion parameters. It creates videos from text prompts with good visual quality and prompt adherence, making advanced video generation accessible to researchers.

## How It Works

**References:**

- Paper: "Allegro: Open the Black Box of Commercial-Level Video Generation Model" (Community, 2024)

Allegro is a ~3B parameter DiT-based video generation model focused on efficiency and quality.
It achieves commercial-level quality through careful architecture design, data curation, and
training strategies while remaining accessible for research. Supports text-to-video generation
with good prompt adherence.

Technical specifications:

- Architecture: Efficient DiT + 3D VAE
- Latent channels: 4
- Default: 88 frames at 15 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AllegroModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of AllegroModel with full customization support. |

