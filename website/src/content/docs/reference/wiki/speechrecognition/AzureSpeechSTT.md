---
title: "AzureSpeechSTT<T>"
description: "Azure Speech Service: Microsoft's enterprise ASR platform"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ProprietaryAPI`

Azure Speech Service: Microsoft's enterprise ASR platform

## For Beginners

Azure Speech Service provides enterprise-grade speech-to-text through Microsoft's cloud platform. The service uses FastTransformer-based models trained on massive multilingual datasets. Features include custom speech models trained on domain data,...

## How It Works

**References:**

- API: "Azure Cognitive Services Speech-to-Text" (Microsoft Azure, 2024)

Azure Speech Service provides enterprise-grade speech-to-text through Microsoft's cloud platform. The service uses FastTransformer-based models trained on massive multilingual datasets. Features include custom speech models trained on domain data, real-time and batch transcription, conversation transcription with speaker diarization, and pronunciation assessment. The custom neural voice feature enables domain adaptation with as little as 30 minutes of training data.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Azure Speech Service architecture. |

