---
title: "Qwen3ASR<T>"
description: "Qwen3-ASR: LLM-integrated ASR with Qwen3 decoder"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

Qwen3-ASR: LLM-integrated ASR with Qwen3 decoder

## For Beginners

Qwen3-ASR integrates a speech encoder with Qwen3's language model for instruction-following ASR. The architecture uses a Conformer audio encoder with a linear adapter projecting into Qwen3's embedding space. The LLM decoder enables flexible output...

## How It Works

**References:**

- Model: "Qwen3-ASR" (Alibaba Qwen, 2025)

Qwen3-ASR integrates a speech encoder with Qwen3's language model for instruction-following ASR. The architecture uses a Conformer audio encoder with a linear adapter projecting into Qwen3's embedding space. The LLM decoder enables flexible output: transcription, translation, summarization, and spoken language understanding via natural language prompts. Supports 50+ languages inheriting Qwen3's multilingual capabilities.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Conformer encoder + Qwen3 LLM decoder. |

