---
title: "VideoPoetModel<T>"
description: "VideoPoet LLM-based zero-shot video generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video`

VideoPoet LLM-based zero-shot video generation.

## For Beginners

VideoPoet from Google uses a large language model (LLM) architecture for zero-shot video generation. The original paper treats video generation as a token prediction task. This implementation approximates its capabilities using a diffusion-based backbone for compatibility with the framework.

## How It Works

**References:**

- Paper: "VideoPoet: A Large Language Model for Zero-Shot Video Generation" (Google, 2024)

VideoPoet uses a large language model for zero-shot video generation, treating video as a
sequence of discrete tokens. The model synthesizes high-quality video with matching audio from
diverse conditioning signals. Won the ICML 2024 award for its approach to video generation.
Note: This implementation uses a DiT-based diffusion approximation of the original LLM architecture.

Technical specifications:

- Architecture: LLM-Based Token Prediction + MAGVIT-v2 Tokenizer
- Latent channels: 16
- Default: 128 frames at 8 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoPoetModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32)` | Initializes a new instance of VideoPoetModel with full customization support. |

