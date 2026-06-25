---
title: "WhisperX<T>"
description: "WhisperX: Whisper with VAD-based segmentation, forced alignment, and speaker diarization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

WhisperX: Whisper with VAD-based segmentation, forced alignment, and speaker diarization.

## For Beginners

WhisperX addresses Whisper's limitations with long-form audio by adding: (1) Voice Activity Detection (VAD) for pre-segmentation, splitting audio at speech boundaries instead of fixed 30s chunks, eliminating hallucination on silence; (2) forced ph...

## How It Works

**References:**

- Paper: "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio" (Bain et al., 2023)

WhisperX addresses Whisper's limitations with long-form audio by adding: (1) Voice Activity
Detection (VAD) for pre-segmentation, splitting audio at speech boundaries instead of fixed
30s chunks, eliminating hallucination on silence; (2) forced phoneme alignment using wav2vec2
for word-level timestamps accurate to ~50ms; (3) speaker diarization via pyannote for speaker
attribution. The pipeline: VAD segmentation -> Whisper transcription -> forced alignment ->
optional diarization. This achieves 12x faster than real-time with accurate word timestamps.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using VAD-segmented Whisper with forced alignment. |

