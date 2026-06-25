---
title: "LanguageIdentifierOptions"
description: "Configuration options for language identification models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.LanguageIdentification`

Configuration options for language identification models.

## For Beginners

Language identification (LID) determines which
language is being spoken in an audio recording.

Key settings:

- SampleRate: Must match your audio (16000 Hz is common for speech)
- MinConfidence: Minimum confidence to report a detection
- TopK: Number of top language predictions to return

Example:

## How It Works

These options configure how language identification models process audio
and return predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LanguageIdentifierOptions` | Initializes a new instance with default values. |
| `LanguageIdentifierOptions(LanguageIdentifierOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets or sets the embedding dimension. |
| `FftSize` | Gets or sets the FFT size for spectrogram computation. |
| `HopLength` | Gets or sets the hop length between frames. |
| `MinConfidence` | Gets or sets the minimum confidence threshold for valid detection. |
| `MinDurationSeconds` | Gets or sets the minimum audio duration in seconds required for reliable detection. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX model options. |
| `SampleRate` | Gets or sets the sample rate of input audio in Hz. |
| `SegmentWindowMs` | Gets or sets the window size in milliseconds for segmented language detection. |
| `TopK` | Gets or sets the number of top language predictions to return. |
| `UseVoiceActivityDetection` | Gets or sets whether to apply voice activity detection before language identification. |

