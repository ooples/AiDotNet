---
title: "IAudioEventDetector<T>"
description: "Interface for audio event detection models that identify specific sounds/events in audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for audio event detection models that identify specific sounds/events in audio.

## For Beginners

Event detection is like having a listener who notes down
every distinct sound they hear and when it happened.

How it works:

1. Audio is analyzed in overlapping windows
2. Each window is classified for the presence of various events
3. Consecutive detections are merged into event segments

Types of events:

- Environmental: Car horn, dog bark, siren, glass breaking
- Speech: Laughter, cough, scream, applause
- Music: Drum hit, guitar strum, piano note
- Industrial: Machine alarm, tool sounds

Use cases:

- Security/surveillance (detect gunshots, breaking glass)
- Smart home (detect doorbell, smoke alarm, baby crying)
- Wildlife monitoring (detect animal calls)
- Content moderation (detect inappropriate sounds)
- Accessibility (alert deaf users to sounds)

Challenges:

- Overlapping events (multiple sounds at once)
- Variable event duration (short beep vs long siren)
- Background noise interference

## How It Works

Audio event detection identifies when specific sounds occur in an audio stream.
Unlike classification which assigns one label to entire clips, event detection
finds multiple events with their timestamps.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportedEvents` | Gets the list of event types this model can detect. |
| `TimeResolution` | Gets the time resolution for event detection in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` | Detects audio events in the audio stream. |
| `Detect(Tensor<>,)` | Detects audio events in the audio stream with custom threshold. |
| `DetectAsync(Tensor<>,CancellationToken)` | Detects audio events asynchronously. |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` | Detects specific events only. |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` | Detects specific events only with custom threshold. |
| `GetEventProbabilities(Tensor<>)` | Gets frame-level event probabilities. |
| `StartStreamingSession` | Performs real-time event detection on a streaming session. |
| `StartStreamingSession(Int32,)` | Performs real-time event detection on a streaming session with custom settings. |

