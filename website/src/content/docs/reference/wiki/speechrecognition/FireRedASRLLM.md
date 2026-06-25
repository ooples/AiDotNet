---
title: "FireRedASRLLM<T>"
description: "FireRedASR-LLM: LLM-enhanced version with Qwen2 decoder"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

FireRedASR-LLM: LLM-enhanced version with Qwen2 decoder

## For Beginners

FireRedASR-LLM extends the base system with Qwen2-7B as the decoder, using the speech encoder's output as prefix tokens for the LLM. This enables the model to leverage the LLM's extensive language knowledge for improved accuracy on complex utteran...

## How It Works

**References:**

- Paper: "FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition" (FireRed Team, 2025)

FireRedASR-LLM extends the base system with Qwen2-7B as the decoder, using the speech encoder's output as prefix tokens for the LLM. This enables the model to leverage the LLM's extensive language knowledge for improved accuracy on complex utterances. The adapter module aligns speech encoder dimensions with the LLM's embedding space. This approach achieves significant WER reductions on hard test sets with proper nouns, code-switching, and domain-specific terminology.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using FireRedASR encoder + Qwen2 LLM decoder. |

