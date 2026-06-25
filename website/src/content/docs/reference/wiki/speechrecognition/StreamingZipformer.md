---
title: "StreamingZipformer<T>"
description: "Streaming Zipformer: multi-scale streaming ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

Streaming Zipformer: multi-scale streaming ASR

## For Beginners

Streaming Zipformer adapts the Zipformer architecture for real-time processing. Zipformer uses temporal downsampling and upsampling between encoder blocks, processing at multiple temporal resolutions simultaneously. Lower-resolution blocks capture...

## How It Works

**References:**

- Paper: "Zipformer: A faster and better encoder for automatic speech recognition" (Yao et al., 2023)

Streaming Zipformer adapts the Zipformer architecture for real-time processing. Zipformer uses temporal downsampling and upsampling between encoder blocks, processing at multiple temporal resolutions simultaneously. Lower-resolution blocks capture long-range dependencies cheaply while high-resolution blocks preserve fine-grained acoustic detail. For streaming, chunk-based processing with causal attention enables real-time operation. Zipformer achieves state-of-the-art speed-accuracy tradeoffs for streaming ASR.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Streaming Zipformer's multi-scale architecture. |

