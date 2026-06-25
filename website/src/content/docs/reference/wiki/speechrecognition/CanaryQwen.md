---
title: "CanaryQwen<T>"
description: "Canary-Qwen: NVIDIA NeMo multilingual ASR + translation with Qwen-2.5 LLM decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

Canary-Qwen: NVIDIA NeMo multilingual ASR + translation with Qwen-2.5 LLM decoder.

## For Beginners

Canary-Qwen combines a Fast Conformer encoder with a Qwen-2.5 1.5B LLM decoder for multilingual ASR and speech translation. The architecture uses a perceiver-style adapter (downsampling cross-attention) between the encoder and the LLM to bridge th...

## How It Works

**References:**

- Model: "Canary-Qwen" (NVIDIA NeMo, 2025)

Canary-Qwen combines a Fast Conformer encoder with a Qwen-2.5 1.5B LLM decoder for
multilingual ASR and speech translation. The architecture uses a perceiver-style adapter
(downsampling cross-attention) between the encoder and the LLM to bridge the modality gap.
Special task tokens (language, translate/transcribe, timestamps) control the output mode.
The model supports 20+ languages for both ASR and any-to-any translation, leveraging the
LLM's pre-trained multilingual text capabilities.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Fast Conformer encoder + Qwen-2.5 LLM decoder. |

