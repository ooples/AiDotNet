---
title: "Kling26Model<T>"
description: "Kling 2.6 with simultaneous audio-visual generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Kling 2.6 with simultaneous audio-visual generation.

## For Beginners

Kling 2.6 generates videos with simultaneous audio and visual output, producing synchronized sound effects and dialogue. It excels at audio-visual coherence where sounds match the visual content naturally.

## How It Works

**References:**

- Reference: Kuaishou Kling 2.6 (2025)

Kling 2.6 delivers fast output with natural sound effects and strong motion consistency.
The model generates synchronized audio-visual content in a single pass, making it particularly
effective for product demonstrations and action sequences. Features advanced depth and cinematic
shot capabilities.

Technical specifications:

- Architecture: DiT + Simultaneous Audio-Visual Generation
- Latent channels: 16
- Default: 120 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Kling26Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of Kling26Model with full customization support. |

