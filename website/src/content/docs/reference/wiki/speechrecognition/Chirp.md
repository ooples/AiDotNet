---
title: "Chirp<T>"
description: "Chirp: Google's production multilingual ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Multilingual`

Chirp: Google's production multilingual ASR

## For Beginners

Chirp is Google's production multilingual ASR model based on the USM architecture, deployed in Google Cloud Speech-to-Text V2. It supports 100+ languages and dialects with a single model. The model uses USM's universal encoder with optimized servi...

## How It Works

**References:**

- Model: "Chirp" (Google Cloud Speech-to-Text V2, 2023)

Chirp is Google's production multilingual ASR model based on the USM architecture, deployed in Google Cloud Speech-to-Text V2. It supports 100+ languages and dialects with a single model. The model uses USM's universal encoder with optimized serving infrastructure for low-latency production deployment. Chirp achieves significant WER reductions over the previous V1 system across all supported languages.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Chirp's production-optimized USM encoder. |

