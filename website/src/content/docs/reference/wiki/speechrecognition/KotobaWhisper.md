---
title: "KotobaWhisper<T>"
description: "Kotoba-Whisper: Japanese-optimized distilled Whisper with 2 decoder layers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

Kotoba-Whisper: Japanese-optimized distilled Whisper with 2 decoder layers.

## For Beginners

Kotoba-Whisper applies Distil-Whisper's knowledge distillation specifically for Japanese ASR. The model retains the full 32-layer encoder from Whisper large-v3 but uses only 2 decoder layers, trained with pseudo-labeled Japanese audio data. Japane...

## How It Works

**References:**

- Model: "Kotoba-Whisper" (Kotoba Technologies, 2024)

Kotoba-Whisper applies Distil-Whisper's knowledge distillation specifically for Japanese ASR.
The model retains the full 32-layer encoder from Whisper large-v3 but uses only 2 decoder
layers, trained with pseudo-labeled Japanese audio data. Japanese-specific optimizations
include: proper handling of kanji/hiragana/katakana tokenization, punctuation normalization
for Japanese text, and CJK-aware text segmentation. The model achieves comparable CER to
the full Whisper large-v3 on Japanese benchmarks while being 6x faster.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the Japanese-optimized distilled encoder-decoder. |

