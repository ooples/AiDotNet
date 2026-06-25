---
title: "SenseVoice<T>"
description: "SenseVoice: multi-task speech understanding model"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

SenseVoice: multi-task speech understanding model

## For Beginners

SenseVoice is a multi-task speech understanding model that handles ASR, language identification, emotion recognition, and audio event detection in a single model. It uses a shared encoder with task-specific output heads. The model processes speech...

## How It Works

**References:**

- Model: "SenseVoice" (Alibaba FunASR, 2024)

SenseVoice is a multi-task speech understanding model that handles ASR, language identification, emotion recognition, and audio event detection in a single model. It uses a shared encoder with task-specific output heads. The model processes speech with a Paraformer-style encoder and uses task tokens to select the output modality. SenseVoice Small covers 50+ languages while maintaining fast inference through non-autoregressive decoding.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using SenseVoice's multi-task encoder with task-specific heads. |

