---
title: "FasterWhisper<T>"
description: "Faster-Whisper: CTranslate2-optimized Whisper with int8/float16 quantization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

Faster-Whisper: CTranslate2-optimized Whisper with int8/float16 quantization.

## For Beginners

Faster-Whisper re-implements Whisper using CTranslate2's optimized inference engine. Key optimizations: (1) int8 quantization reducing model size by ~4x with minimal accuracy loss; (2) batched beam search with KV-cache reuse; (3) fused attention k...

## How It Works

**References:**

- Implementation: "Faster-Whisper" (SYSTRAN/CTranslate2, 2023)

Faster-Whisper re-implements Whisper using CTranslate2's optimized inference engine.
Key optimizations: (1) int8 quantization reducing model size by ~4x with minimal accuracy loss;
(2) batched beam search with KV-cache reuse; (3) fused attention kernels; (4) efficient
memory management. The model achieves ~4x faster than the original OpenAI implementation
while using less memory. Supports the same encoder-decoder architecture as Whisper with
configurable compute types (int8, float16, float32) for speed/accuracy tradeoffs.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the CTranslate2-optimized encoder-decoder pipeline. |

