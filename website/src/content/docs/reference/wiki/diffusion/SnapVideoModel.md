---
title: "SnapVideoModel<T>"
description: "Snap Video scaled spatiotemporal transformer for text-to-video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.LongVideo`

Snap Video scaled spatiotemporal transformer for text-to-video.

## For Beginners

SnapVideo from Snap uses a scaled spatiotemporal transformer that processes space and time jointly for high-quality video generation. Its efficient architecture scales well to longer durations.

## How It Works

**References:**

- Paper: "Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis" (Snap, 2024)

Snap Video uses a scaled spatiotemporal transformer architecture for efficient text-to-video
synthesis. The model scales transformer attention across both spatial and temporal dimensions
with efficient factorization strategies. Achieves strong quality with efficient generation.

Technical specifications:

- Architecture: Scaled Spatiotemporal Transformer
- Latent channels: 16
- Default: 48 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SnapVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of SnapVideoModel with full customization support. |

