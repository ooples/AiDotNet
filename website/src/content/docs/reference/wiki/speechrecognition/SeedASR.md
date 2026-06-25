---
title: "SeedASR<T>"
description: "Seed-ASR: ByteDance's large-scale multilingual ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

Seed-ASR: ByteDance's large-scale multilingual ASR

## For Beginners

Seed-ASR is ByteDance's large-scale ASR system that uses an LLM decoder to understand diverse speech and contexts. The system processes audio through a Conformer encoder and feeds representations into a large language model via an audio-text adapt...

## How It Works

**References:**

- Paper: "Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition" (ByteDance, 2024)

Seed-ASR is ByteDance's large-scale ASR system that uses an LLM decoder to understand diverse speech and contexts. The system processes audio through a Conformer encoder and feeds representations into a large language model via an audio-text adapter. The LLM's broad knowledge enables it to handle proper nouns, code-switching, and domain-specific content. Seed-ASR supports multilingual transcription and achieves state-of-the-art results across Chinese and English benchmarks.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Seed-ASR's encoder + LLM decoder architecture. |

