---
title: "CosmosModel<T>"
description: "NVIDIA Cosmos physics-aware world generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.WorldModels`

NVIDIA Cosmos physics-aware world generation model.

## For Beginners

Cosmos from NVIDIA is a physics-aware world generation model that understands real-world physics. It comes in Nano/Super/Ultra sizes and generates videos where objects obey gravity, collisions, and other physical laws.

## How It Works

**References:**

- Reference: NVIDIA Cosmos (2025)

Cosmos is NVIDIA's world foundation model for physical AI development. Split into Nano (4B),
Super (8B), and Ultra (14B) tiers, it generates physics-aware video and synthetic sensor data.
Cosmos Predict 2.5 unifies Text2World, Image2World, and Video2World into a single architecture
for consistent, controllable multi-camera video worlds.

Technical specifications:

- Architecture: Physics-Aware DiT + World Generation + Multi-Camera
- Latent channels: 16
- Default: 120 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosmosModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of CosmosModel with full customization support. |

