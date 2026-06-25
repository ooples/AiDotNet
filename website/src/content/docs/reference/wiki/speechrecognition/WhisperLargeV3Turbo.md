---
title: "WhisperLargeV3Turbo<T>"
description: "Whisper large-v3-turbo: OpenAI's 809M parameter distilled Whisper with only 4 decoder layers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

Whisper large-v3-turbo: OpenAI's 809M parameter distilled Whisper with only 4 decoder layers.

## For Beginners

Whisper large-v3-turbo retains the full 32-layer encoder from large-v3 but reduces the decoder from 32 layers to only 4, achieving ~6x faster inference with minimal quality loss. The model uses 128 mel bins, 20-head attention, and the same 51866-t...

## How It Works

**References:**

- Technical note: "Whisper large-v3-turbo" (OpenAI, 2024)

Whisper large-v3-turbo retains the full 32-layer encoder from large-v3 but reduces the
decoder from 32 layers to only 4, achieving ~6x faster inference with minimal quality loss.
The model uses 128 mel bins, 20-head attention, and the same 51866-token multilingual
vocabulary. The key insight is that the encoder does the heavy lifting for speech
understanding, while a much smaller decoder suffices for autoregressive text generation.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the turbo encoder-decoder pipeline. |

