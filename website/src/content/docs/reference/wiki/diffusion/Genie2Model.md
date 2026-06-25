---
title: "Genie2Model<T>"
description: "Genie 2 real-time interactive 3D environment generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.WorldModels`

Genie 2 real-time interactive 3D environment generation.

## For Beginners

Genie 2 from Google DeepMind generates real-time interactive 3D environments from a single image. It creates explorable worlds where user actions produce consistent, realistic responses.

## How It Works

**References:**

- Reference: Google DeepMind Genie 2 (2024)

Genie 2 generates real-time interactive 3D environments from single images or text descriptions.
The model enables interactive exploration of generated worlds with consistent physics and
persistent state. Users can navigate and interact with the generated environment in real-time.

Technical specifications:

- Architecture: Action-Conditioned DiT + Interactive 3D + Real-Time
- Latent channels: 16
- Default: 60 frames at 30 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Genie2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of Genie2Model with full customization support. |

