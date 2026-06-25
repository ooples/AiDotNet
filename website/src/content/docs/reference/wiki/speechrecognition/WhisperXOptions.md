---
title: "WhisperXOptions"
description: "Options for WhisperX (Bain et al., 2023): Whisper + forced alignment + VAD + diarization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SpeechRecognition.WhisperFamily`

Options for WhisperX (Bain et al., 2023): Whisper + forced alignment + VAD + diarization.

## For Beginners

These options configure the WhisperX model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WhisperXOptions` | Initializes a new instance with default values. |
| `WhisperXOptions(WhisperXOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableDiarization` | Whether to enable speaker diarization. |
| `VadMinSilenceDuration` | Minimum silence duration in seconds for VAD segmentation. |
| `VadMinSpeechDuration` | Minimum speech segment duration in seconds for VAD. |

