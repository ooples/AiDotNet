---
title: "FasterWhisperOptions"
description: "Options for Faster-Whisper (SYSTRAN/CTranslate2, 2023): CTranslate2-optimized Whisper with int8 quantization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SpeechRecognition.WhisperFamily`

Options for Faster-Whisper (SYSTRAN/CTranslate2, 2023): CTranslate2-optimized Whisper with int8 quantization.

## For Beginners

These options configure the FasterWhisper model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FasterWhisperOptions` | Initializes a new instance with default values. |
| `FasterWhisperOptions(FasterWhisperOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BeamSize` | Number of parallel transcription beams. |
| `ComputeType` | Compute type for quantization (int8, float16, float32). |

