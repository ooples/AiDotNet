---
title: "GameGenXModel<T>"
description: "GameGen-X open-world game video generation with interactive control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.WorldModels`

GameGen-X open-world game video generation with interactive control.

## For Beginners

GameGen-X generates open-world game environments with interactive control. Users can take actions (move, interact) and the model generates the next frame in response, creating a playable game experience.

## How It Works

**References:**

- Paper: "GameGen-X: Interactive Open-world Game Video Generation" (ICLR 2025)

GameGen-X is the first open-world game generation model supporting interactive control. Through
a two-stage training strategy, it achieves high-quality game content generation and dynamic
control, with significant advances in character diversity, environmental interaction, and event
simulation. Accepted at ICLR 2025.

Technical specifications:

- Architecture: DiT + Interactive Control + Two-Stage Training
- Latent channels: 16
- Default: 60 frames at 30 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GameGenXModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of GameGenXModel with full customization support. |

