---
title: "StepVideoModel<T>"
description: "StepVideo text-to-video model with benchmark-leading quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

StepVideo text-to-video model with benchmark-leading quality.

## For Beginners

Step Video is a ~10B parameter text-to-video model that competes with state-of-the-art proprietary models. It excels at generating diverse, high-quality video content with strong prompt adherence.

## How It Works

**References:**

- Paper: "StepVideo: Text-to-Video Generation with Scalable Diffusion Transformers" (StepFun, 2025)

StepVideo is a ~10B parameter T2V model from StepFun competing with HunyuanVideo on quality
benchmarks. It uses a scalable DiT architecture with deep text understanding, handling complex
prompts and quality motion reliably. The model is released as open-source with weights and code.

Technical specifications:

- Architecture: Scalable DiT + Deep Text Understanding
- Latent channels: 16
- Default: 51 frames at 16 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StepVideoModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of StepVideoModel with full customization support. |

