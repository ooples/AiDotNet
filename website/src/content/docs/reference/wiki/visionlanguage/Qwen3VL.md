---
title: "Qwen3VL<T>"
description: "Qwen3-VL: latest series with 2B/4B/8B/32B variants and cross-attention resampler."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Qwen3-VL: latest series with 2B/4B/8B/32B variants and cross-attention resampler.

## For Beginners

Qwen3-VL is the latest generation of Alibaba's Qwen vision-language
models, available in 2B, 4B, 8B, and 32B sizes to fit different compute budgets. It builds
on the innovations from Qwen2-VL (dynamic resolution, M-RoPE) with improved training data
and model optimization using the Qwen3 language backbone. The cross-attention resampler
efficiently compresses visual tokens before feeding them to the language model, keeping
inference fast even for high-resolution images. The range of sizes means you can pick the
right trade-off between performance and cost for your application. Default values follow
the original paper settings.

## How It Works

Qwen3-VL is the latest generation of the Qwen vision-language model series, available in
multiple sizes (2B, 4B, 8B, 32B). It inherits the cross-attention resampler architecture
from Qwen2-VL with improvements in training data and model optimization, using Qwen3 as
the language backbone.

**References:**

- Paper: "Qwen3-VL Technical Report" (2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Qwen3-VL's cross-attention resampler architecture. |
| `GetExtraTrainableLayers` |  |

