---
title: "VoiceActivityDetectorBase<T>"
description: "Base class for algorithmic voice activity detection implementations (non-neural network)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.VoiceActivity`

Base class for algorithmic voice activity detection implementations (non-neural network).

## For Beginners

VAD answers a simple question: "Is someone speaking right now?"

Common uses:

- Skip silence during transcription
- Reduce transmission bandwidth in VoIP
- Trigger recording only when speech is detected
- Segment audio into speaker turns

This base class provides:

- Frame-based processing with hangover logic
- Streaming mode with state management
- Segment detection across entire audio files

For neural network-based VAD (like Silero), see classes that extend AudioNeuralNetworkBase.

## How It Works

Voice Activity Detection (VAD) determines whether audio contains speech or silence.
This is fundamental to many audio applications including speech recognition,
communication systems, and noise reduction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoiceActivityDetectorBase(Int32,Int32,Double,Int32,Int32)` | Initializes a new instance of VoiceActivityDetectorBase. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `FrameSize` |  |
| `MinSilenceDurationMs` |  |
| `MinSpeechDurationMs` |  |
| `SampleRate` |  |
| `Threshold` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFrameProbability([])` | Computes speech probability for a single frame. |
| `DetectSpeech(Tensor<>)` |  |
| `DetectSpeechSegments(Tensor<>)` |  |
| `GetFrameProbabilities(Tensor<>)` |  |
| `GetSpeechProbability(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `_inSpeech` | Current speech state. |
| `_silenceFrameCount` | Number of consecutive silence frames. |
| `_speechFrameCount` | Number of consecutive speech frames. |

