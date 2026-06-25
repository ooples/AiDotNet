---
title: "LumaRay3Model<T>"
description: "Luma Ray 3 with Hi-Fi Diffusion for 4K HDR video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Luma Ray 3 with Hi-Fi Diffusion for 4K HDR video.

## For Beginners

Luma Ray 3 uses Hi-Fi Diffusion to generate videos at up to 4K HDR quality. It produces the highest resolution output among video generation models with exceptional visual clarity.

## How It Works

**References:**

- Reference: Luma AI Ray 3 (2025)

Ray 3 introduces Hi-Fi Diffusion technology that produces production-ready high-fidelity 4K HDR
footage. The model packs significantly more detail in the same resolution compared to Ray 2,
producing crisp outputs with state-of-the-art realism, physics, and character consistency.

Technical specifications:

- Architecture: Hi-Fi Diffusion + 4K HDR + DiT
- Latent channels: 16
- Default: 120 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LumaRay3Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of LumaRay3Model with full customization support. |

