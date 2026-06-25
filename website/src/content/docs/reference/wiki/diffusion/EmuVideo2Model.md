---
title: "EmuVideo2Model<T>"
description: "Emu Video 2 with improved generation quality and motion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.AudioVisual`

Emu Video 2 with improved generation quality and motion.

## For Beginners

Emu Video 2 improves on the original with better generation quality, sharper details, and more natural motion. It builds on Meta's Emu architecture with enhanced temporal modeling.

## How It Works

**References:**

- Reference: Meta Emu Video 2 (2024)

Emu Video 2 improves upon the original with enhanced generation quality, better motion dynamics,
and longer video support. Uses an improved DiT backbone with refined temporal attention and
better conditioning mechanisms for more natural and diverse video generation.

Technical specifications:

- Architecture: Improved Factorized T2V with Enhanced Temporal DiT
- Latent channels: 16
- Default: 32 frames at 16 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EmuVideo2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of EmuVideo2Model with full customization support. |

