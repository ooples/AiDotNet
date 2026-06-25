---
title: "DIAMONDModel<T>"
description: "DIAMOND diffusion-based game engine from video with action conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.WorldModels`

DIAMOND diffusion-based game engine from video with action conditioning.

## For Beginners

DIAMOND is a game engine built from video - it learns to simulate game environments from recorded gameplay. Given an action (e.g., press right), it predicts what the next frame should look like.

## How It Works

**References:**

- Paper: "DIAMOND: Diffusion for World Modeling" (2024)

DIAMOND (DIffusion As a Model Of eNvironment Dreams) creates a game engine from video using
action-conditioned diffusion. The model learns environment dynamics from gameplay video and
can generate consistent game frames conditioned on player actions, effectively creating a
playable world model.

Technical specifications:

- Architecture: Action-Conditioned Diffusion + World Dynamics Learning
- Latent channels: 16
- Default: 1 frames at 30 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DIAMONDModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of DIAMONDModel with full customization support. |

