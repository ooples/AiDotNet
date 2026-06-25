---
title: "NVLM<T>"
description: "NVLM 1.0: NVIDIA's cross-attention + decoder-only hybrid VLM that retains text performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

NVLM 1.0: NVIDIA's cross-attention + decoder-only hybrid VLM that retains text performance.

## For Beginners

NVLM from NVIDIA takes a hybrid approach that combines two
ways of processing visual information: cross-attention (where the language model attends
to visual features through dedicated cross-attention layers) and decoder-only (where image
tokens are simply concatenated with text tokens). Most models use one or the other, but
NVLM uses both, which lets it retain strong text-only performance while adding powerful
vision capabilities. This is important because many multimodal models lose some text
ability when adding vision — NVLM avoids this trade-off. Default values follow the
original paper settings.

## How It Works

NVLM 1.0 (NVIDIA, 2024) introduces a hybrid architecture combining cross-attention and
decoder-only approaches. This design allows the model to process visual features via both
cross-attention layers and direct token concatenation, retaining strong text-only performance.

**References:**

- Paper: "NVLM: Open Frontier-Class Multimodal LLMs" (NVIDIA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using NVLM's cross-attention + decoder-only hybrid architecture. |

