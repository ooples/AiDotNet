---
title: "SenseVoiceLarge<T>"
description: "SenseVoice-Large: scaled multi-task speech model"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

SenseVoice-Large: scaled multi-task speech model

## For Beginners

SenseVoice-Large scales up the SenseVoice architecture with a larger encoder (1024-dim, 50 layers) for improved accuracy across all supported tasks. The model supports 50+ languages and includes enhanced emotion recognition and audio event detecti...

## How It Works

**References:**

- Model: "SenseVoice-Large" (Alibaba FunASR, 2024)

SenseVoice-Large scales up the SenseVoice architecture with a larger encoder (1024-dim, 50 layers) for improved accuracy across all supported tasks. The model supports 50+ languages and includes enhanced emotion recognition and audio event detection capabilities. Uses Whisper-style 128 mel bins for better frequency resolution.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the scaled SenseVoice-Large multi-task pipeline. |

