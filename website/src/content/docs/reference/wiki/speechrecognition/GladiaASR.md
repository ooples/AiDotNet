---
title: "GladiaASR<T>"
description: "Gladia: enterprise audio transcription with Whisper"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ProprietaryAPI`

Gladia: enterprise audio transcription with Whisper

## For Beginners

Gladia provides enterprise audio transcription powered by optimized Whisper models with proprietary enhancements. The service adds real-time streaming, word-level timestamps, speaker diarization, and audio intelligence features on top of Whisper's...

## How It Works

**References:**

- API: "Gladia Audio Transcription" (Gladia, 2024)

Gladia provides enterprise audio transcription powered by optimized Whisper models with proprietary enhancements. The service adds real-time streaming, word-level timestamps, speaker diarization, and audio intelligence features on top of Whisper's multilingual capabilities. Gladia's proprietary preprocessing pipeline handles audio normalization, VAD-based segmentation, and noise reduction before feeding audio to the transcription model. The API supports 100+ languages with automatic language detection.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Gladia's enhanced Whisper-based architecture. |

