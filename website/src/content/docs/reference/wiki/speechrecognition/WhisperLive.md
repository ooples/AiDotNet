---
title: "WhisperLive<T>"
description: "WhisperLive: Real-time streaming transcription using Whisper with VAD-based chunking."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

WhisperLive: Real-time streaming transcription using Whisper with VAD-based chunking.

## For Beginners

WhisperLive enables real-time streaming transcription by combining Whisper with Voice Activity Detection (VAD). Audio is chunked based on speech boundaries rather than fixed intervals, allowing natural sentence-level transcription. The system uses...

## How It Works

**References:**

- Implementation: "WhisperLive: Real-Time Whisper Transcription" (Collabora, 2024)

WhisperLive enables real-time streaming transcription by combining Whisper with Voice
Activity Detection (VAD). Audio is chunked based on speech boundaries rather than fixed
intervals, allowing natural sentence-level transcription. The system uses a client-server
architecture with WebSocket streaming. The encoder processes each VAD-segmented chunk
independently, and the decoder generates text with local context. Optimizations include
TensorRT/CTranslate2 backends and adaptive chunk sizing based on speech patterns.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using VAD-chunked streaming Whisper. |

