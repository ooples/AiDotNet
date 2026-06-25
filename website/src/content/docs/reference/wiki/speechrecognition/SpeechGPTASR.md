---
title: "SpeechGPTASR<T>"
description: "SpeechGPT: speech-language model for multi-modal conversation"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

SpeechGPT: speech-language model for multi-modal conversation

## For Beginners

SpeechGPT extends a large language model with intrinsic speech understanding and generation capabilities. For ASR, speech is discretized into tokens using HuBERT + k-means clustering, then these discrete speech tokens are treated as a new modality...

## How It Works

**References:**

- Paper: "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities" (Zhang et al., Fudan, 2023)

SpeechGPT extends a large language model with intrinsic speech understanding and generation capabilities. For ASR, speech is discretized into tokens using HuBERT + k-means clustering, then these discrete speech tokens are treated as a new modality in the LLM's vocabulary. The model learns to translate between speech tokens and text tokens through instruction tuning. This enables seamless multi-modal conversation without separate encoder/decoder components.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using SpeechGPT's discrete speech token approach. |

