---
title: "IStreamingTts<T>"
description: "Interface for TTS models that support streaming/chunked synthesis with low latency."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TextToSpeech.Interfaces`

Interface for TTS models that support streaming/chunked synthesis with low latency.

## How It Works

Streaming TTS models can begin outputting audio before the full utterance is processed,
enabling low-latency applications like conversational AI:

- CosyVoice 2: 150ms first-packet latency with streaming flow matching
- Chatterbox: real-time streaming with emotion control
- XTTS-v2: chunked streaming for voice cloning

## Properties

| Property | Summary |
|:-----|:--------|
| `FirstPacketLatencyMs` | Gets the target first-packet latency in milliseconds. |
| `HasMoreChunks` | Gets whether there are more audio chunks available. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SynthesizeFirstChunk(String,Int32)` | Synthesizes audio in streaming chunks, returning the first available audio chunk. |
| `SynthesizeNextChunk` | Gets the next audio chunk from an ongoing streaming synthesis. |

