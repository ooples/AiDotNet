---
title: "RealisVSR<T>"
description: "RealisVSR: detail-enhanced diffusion for real-world 4K video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

RealisVSR: detail-enhanced diffusion for real-world 4K video super-resolution.

## For Beginners

RealisVSR takes your low-quality video and uses a powerful AI
video generator to create a high-quality 4K version. A special "detail enhancer"
makes sure that small details like text on signs and fine textures don't get lost
or hallucinated during the upscaling process.

**Usage:**

## How It Works

RealisVSR (2025) uses the Wan 2.1 video diffusion backbone with detail enhancement:

- Wan 2.1 backbone: pretrained text-to-video diffusion model provides strong video priors
- ControlNet detail adapter: a trainable copy of the encoder conditions the generation on

low-resolution input while preserving fine-grained spatial details (text, edges, textures)

- Motion-aware temporal conditioning: ensures consistent motion across generated frames
- Designed for upscaling real-world degraded video to 4K resolution

**Reference:** "RealisVSR: Detail-Enhanced Diffusion for Real-World 4K Video
Super-Resolution" (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealisVSR(NeuralNetworkArchitecture<>,RealisVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a RealisVSR model in native training mode. |
| `RealisVSR(NeuralNetworkArchitecture<>,String,RealisVSROptions)` | Creates a RealisVSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

