---
title: "SALM<T>"
description: "SALM: speech-augmented language model for ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

SALM: speech-augmented language model for ASR

## For Beginners

SALM augments an LLM with speech understanding via in-context learning. A frozen speech encoder (Conformer or wav2vec 2.0) extracts features, which are projected into the LLM's embedding space via a lightweight adapter. The LLM generates transcrip...

## How It Works

**References:**

- Paper: "SALM: Speech-augmented Language Model with In-context Learning for Speech Recognition" (NVIDIA, 2024)

SALM augments an LLM with speech understanding via in-context learning. A frozen speech encoder (Conformer or wav2vec 2.0) extracts features, which are projected into the LLM's embedding space via a lightweight adapter. The LLM generates transcriptions using in-context examples and instructions. SALM demonstrates that LLMs can perform competitive ASR through prompting strategies without explicit ASR fine-tuning, leveraging their pre-trained language abilities.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using SALM's in-context learning approach. |

