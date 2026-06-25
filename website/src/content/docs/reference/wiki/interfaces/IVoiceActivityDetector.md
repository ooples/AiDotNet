---
title: "IVoiceActivityDetector<T>"
description: "Defines the contract for Voice Activity Detection (VAD) models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for Voice Activity Detection (VAD) models.

## For Beginners

VAD answers the question "Is someone speaking right now?"

Why VAD is important:

- Speech Recognition: Only process audio when speech is present (saves compute)
- Voice Assistants: Detect when user starts/stops talking
- VoIP/Video Calls: Only transmit audio when speaking (saves bandwidth)
- Transcription: Find speech segments in long recordings
- Speaker Diarization: First step to identify who spoke when

How it works:

1. Traditional: Look at energy levels, zero-crossing rate, spectral features
2. Modern (Neural): Train a model to classify frames as speech/non-speech

Key metrics:

- Accuracy: How often it's correct
- False Positive Rate: Saying "speech" when it's noise (annoying in voice assistants)
- False Negative Rate: Missing actual speech (drops words in transcription)
- Latency: How quickly it detects speech onset

## How It Works

Voice Activity Detection determines when speech is present in an audio signal.
This is a fundamental building block for many speech processing systems.

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameSize` | Gets the frame size in samples used for detection. |
| `MinSilenceDurationMs` | Gets or sets the minimum silence duration in milliseconds. |
| `MinSpeechDurationMs` | Gets or sets the minimum speech duration in milliseconds. |
| `SampleRate` | Gets the sample rate this VAD operates at. |
| `Threshold` | Gets or sets the detection threshold (0.0 to 1.0). |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectSpeech(Tensor<>)` | Detects whether speech is present in an audio frame. |
| `DetectSpeechSegments(Tensor<>)` | Detects speech segments in a longer audio recording. |
| `GetFrameProbabilities(Tensor<>)` | Gets frame-by-frame speech probabilities for the entire audio. |
| `GetSpeechProbability(Tensor<>)` | Gets the speech probability for an audio frame. |
| `ProcessChunk(Tensor<>)` | Processes audio in streaming mode, maintaining state between calls. |
| `ResetState` | Resets internal state for streaming mode. |

