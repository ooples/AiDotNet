---
title: "KyutaiMoshi<T>"
description: "Kyutai Moshi: full-duplex spoken dialogue model"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

Kyutai Moshi: full-duplex spoken dialogue model

## For Beginners

Moshi is a full-duplex speech-text foundation model that can simultaneously listen and speak, enabling natural spoken dialogue. For ASR, Moshi uses a Mimi neural audio codec to discretize speech into tokens at multiple temporal resolutions. A Tran...

## How It Works

**References:**

- Paper: "Moshi: a speech-text foundation model for real-time dialogue" (Kyutai, 2024)

Moshi is a full-duplex speech-text foundation model that can simultaneously listen and speak, enabling natural spoken dialogue. For ASR, Moshi uses a Mimi neural audio codec to discretize speech into tokens at multiple temporal resolutions. A Transformer backbone processes both the codec tokens and text tokens in an interleaved fashion. The model can transcribe speech while generating responses, achieving real-time spoken conversation with 200ms turn-taking latency.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Moshi's neural codec + Transformer architecture. |

