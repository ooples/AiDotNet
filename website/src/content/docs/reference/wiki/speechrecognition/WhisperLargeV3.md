---
title: "WhisperLargeV3<T>"
description: "Whisper large-v3: OpenAI's 1.55B parameter multilingual encoder-decoder ASR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

Whisper large-v3: OpenAI's 1.55B parameter multilingual encoder-decoder ASR.

## For Beginners

Whisper large-v3 uses a 32-layer Transformer encoder-decoder trained on 680k hours of multilingual audio. Key improvements over v2: 128 mel bins (vs 80), improved language coverage (100 languages), and better long-form transcription. The encoder p...

## How It Works

**References:**

- Paper: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., OpenAI, 2023)

Whisper large-v3 uses a 32-layer Transformer encoder-decoder trained on 680k hours of
multilingual audio. Key improvements over v2: 128 mel bins (vs 80), improved language
coverage (100 languages), and better long-form transcription. The encoder processes
log-mel spectrograms via two conv layers + positional embeddings, then N Transformer blocks.
The decoder autoregressively generates text tokens with cross-attention to the encoder.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Whisper's encoder-decoder pipeline. |

