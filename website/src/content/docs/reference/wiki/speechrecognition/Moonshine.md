---
title: "Moonshine<T>"
description: "Moonshine: efficient real-time streaming ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

Moonshine: efficient real-time streaming ASR

## For Beginners

Moonshine is designed for real-time on-device speech recognition with minimal latency. It uses a compact encoder-decoder architecture with rotary position embeddings (RoPE) instead of absolute positional encodings, enabling efficient processing of...

## How It Works

**References:**

- Paper: "Moonshine: Speech Recognition for Live Transcription and Voice Commands" (Useful Sensors, 2024)

Moonshine is designed for real-time on-device speech recognition with minimal latency. It uses a compact encoder-decoder architecture with rotary position embeddings (RoPE) instead of absolute positional encodings, enabling efficient processing of variable-length audio. The encoder processes fixed-size audio chunks while the decoder generates text with cross-attention. Moonshine achieves Whisper-level accuracy at 5x faster inference speed, running in real-time on a Raspberry Pi 4.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Moonshine's efficient encoder-decoder architecture. |

