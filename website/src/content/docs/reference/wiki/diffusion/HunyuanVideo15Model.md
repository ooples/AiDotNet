---
title: "HunyuanVideo15Model<T>"
description: "HunyuanVideo 1.5 efficient video generation model for consumer GPUs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

HunyuanVideo 1.5 efficient video generation model for consumer GPUs.

## For Beginners

HunyuanVideo 1.5 is a large (8.3B parameter) video generator that can run on consumer GPUs. It combines image and video generation in one model, understands complex text prompts via a multimodal language model encoder, and produces high-quality clips at up to 720p.

## How It Works

**References:**

- Paper: "HunyuanVideo 1.5: A Systematic Framework for Large Video Generation Model" (Tencent, 2025)

HunyuanVideo 1.5 is an efficient 8.3B parameter model achieving state-of-the-art visual quality
while running on consumer GPUs. It combines a unified image-video generation architecture with
a Causal 3D VAE and dual-stream DiT blocks. The model uses MLLM text encoding for precise
prompt understanding and supports both T2V and I2V.

Technical specifications:

- Architecture: Dual-Stream DiT + Causal 3D VAE + MLLM Text Encoder
- Latent channels: 16
- Default: 49 frames at 24 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HunyuanVideo15Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of HunyuanVideo15Model with full customization support. |

