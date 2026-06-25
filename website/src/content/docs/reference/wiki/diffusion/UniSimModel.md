---
title: "UniSimModel<T>"
description: "UniSim universal simulator from video and action pairs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.WorldModels`

UniSim universal simulator from video and action pairs.

## For Beginners

UniSim is a universal simulator that learns to simulate any environment from video and action pairs. It generalizes across different domains (robotics, games, real-world) to predict future observations.

## How It Works

**References:**

- Paper: "UniSim: Learning Interactive Real-World Simulators" (2024)

UniSim is a universal simulator that learns from diverse real-world and simulated videos.
Given a previous video segment and an action prompt, it predicts the continuation video through
supervised learning. The model enables simulation of diverse real-world interactions for
robotics, gaming, and embodied AI applications.

Technical specifications:

- Architecture: Universal Simulation DiT + Action-Video Pairs
- Latent channels: 16
- Default: 30 frames at 15 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniSimModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of UniSimModel with full customization support. |

