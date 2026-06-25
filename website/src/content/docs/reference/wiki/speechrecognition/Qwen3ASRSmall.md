---
title: "Qwen3ASRSmall<T>"
description: "Qwen3-ASR-Small: lightweight LLM ASR variant"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

Qwen3-ASR-Small: lightweight LLM ASR variant

## For Beginners

Qwen3-ASR-Small is a compact variant using a smaller encoder (512-dim, 12 layers) paired with Qwen3-0.5B for efficient deployment. Maintains multilingual support and instruction-following capabilities while targeting edge and mobile platforms with...

## How It Works

**References:**

- Model: "Qwen3-ASR-Small" (Alibaba Qwen, 2025)

Qwen3-ASR-Small is a compact variant using a smaller encoder (512-dim, 12 layers) paired with Qwen3-0.5B for efficient deployment. Maintains multilingual support and instruction-following capabilities while targeting edge and mobile platforms with reduced compute requirements.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using compact Conformer encoder + Qwen3-0.5B decoder. |

