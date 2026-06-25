---
title: "MoonshineBase<T>"
description: "Moonshine Base: tiny model for edge deployment"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

Moonshine Base: tiny model for edge deployment

## For Beginners

Moonshine Base is the smallest variant of the Moonshine family at 27M parameters, optimized for edge deployment on resource-constrained devices. The model uses a 4-layer encoder and 4-layer decoder with 288-dim embeddings. Despite its small size, ...

## How It Works

**References:**

- Model: "Moonshine Base" (Useful Sensors, 2024)

Moonshine Base is the smallest variant of the Moonshine family at 27M parameters, optimized for edge deployment on resource-constrained devices. The model uses a 4-layer encoder and 4-layer decoder with 288-dim embeddings. Despite its small size, Moonshine Base achieves usable accuracy for voice commands and short-form transcription, running at 10x real-time on Raspberry Pi 4 and supporting on-device wake word + transcription pipelines.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Moonshine Base's compact encoder-decoder. |

