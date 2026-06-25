---
title: "Mochi1PreviewModel<T>"
description: "Mochi 1 Preview with Asymmetric Diffusion Transformer (AsymmDiT)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

Mochi 1 Preview with Asymmetric Diffusion Transformer (AsymmDiT).

## For Beginners

Mochi 1 Preview uses an Asymmetric Diffusion Transformer (AsymmDiT) that processes text and video with different-sized networks. This asymmetry efficiently handles rich text understanding for prompt adherence while maintaining high video quality.

## How It Works

**References:**

- Paper: "Mochi 1: A New SOTA in Open-Source Video Generation" (Genmo, 2024)

Mochi 1 Preview is a 10B parameter model using the novel Asymmetric Diffusion Transformer
(AsymmDiT) architecture, which uses different computational blocks for the conditioning
and generation pathways. This asymmetry enables efficient processing of text conditions while
maximizing video generation quality. Generates 480p video at 30 FPS.

Technical specifications:

- Architecture: AsymmDiT (Asymmetric Diffusion Transformer)
- Latent channels: 12
- Default: 163 frames at 30 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mochi1PreviewModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of Mochi1PreviewModel with full customization support. |

