---
title: "GraniteSpeech<T>"
description: "Granite Speech: IBM's enterprise speech-language model"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

Granite Speech: IBM's enterprise speech-language model

## For Beginners

Granite Speech integrates speech understanding into IBM's Granite enterprise LLM family. A pre-trained speech encoder (based on Conformer) is paired with a lightweight adapter that maps speech representations to the Granite LLM's input space. The ...

## How It Works

**References:**

- Paper: "Granite Speech: Integrating Speech into Enterprise LLMs" (IBM Research, 2025)

Granite Speech integrates speech understanding into IBM's Granite enterprise LLM family. A pre-trained speech encoder (based on Conformer) is paired with a lightweight adapter that maps speech representations to the Granite LLM's input space. The model supports multi-turn spoken dialogue, speech translation, and domain-specific ASR with enterprise-grade accuracy. Fine-tuning on domain data enables specialized performance for business applications.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Granite's speech encoder + enterprise LLM decoder. |

