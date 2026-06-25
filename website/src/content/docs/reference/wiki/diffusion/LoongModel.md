---
title: "LoongModel<T>"
description: "Loong autoregressive LLM-based minute-long video generator."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Video.LongVideo`

Loong autoregressive LLM-based minute-long video generator.

## For Beginners

Loong uses a large language model approach to generate minute-long videos by treating video frames as tokens in a sequence. This autoregressive approach naturally handles long-range temporal dependencies.

## How It Works

**References:**

- Paper: "Loong: Generating Minute-level Long Videos with Autoregressive Language Models" (2024)

Loong generates minute-long videos using an autoregressive LLM-based approach with progressive
short-to-long training. It addresses the loss imbalance problem for long video training through
a novel loss re-weighting scheme. The model treats video generation as next-token prediction
over visual tokens.

Technical specifications:

- Architecture: Autoregressive LLM + Progressive Short-to-Long Training
- Latent channels: 16
- Default: 480 frames at 8 FPS
- Supports I2V: Yes | T2V: Yes | V2V: No

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoongModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,TemporalVAE<>,IConditioningModule<>,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of LoongModel with full customization support. |

