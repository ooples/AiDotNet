---
title: "StreamingConformer<T>"
description: "Streaming Conformer: chunk-based Conformer with lookahead"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

Streaming Conformer: chunk-based Conformer with lookahead

## For Beginners

Streaming Conformer adapts the standard Conformer architecture for streaming inference by processing audio in fixed-size chunks with limited right-context lookahead. Causal convolutions replace standard convolutions, and attention is restricted to...

## How It Works

**References:**

- Paper: "Conformer: Convolution-augmented Transformer for End-to-End Speech Recognition" (Gulati et al., 2020)

Streaming Conformer adapts the standard Conformer architecture for streaming inference by processing audio in fixed-size chunks with limited right-context lookahead. Causal convolutions replace standard convolutions, and attention is restricted to the current chunk plus a small lookahead window. A chunk-wise cache mechanism stores key-value pairs from previous chunks, enabling the model to maintain long-range context while processing audio in real-time with controllable latency.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using chunk-based Conformer with limited lookahead. |

