---
title: "AudioBuilderExtensions"
description: "Audio event detection and classification extensions for AiModelBuilder."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet`

Audio event detection and classification extensions for AiModelBuilder.

## For Beginners

Audio event detection identifies sounds in audio recordings
or real-time streams. After configuring a model (like BEATs or AudioEventDetector),
use these methods to detect sounds, get probabilities, or start real-time monitoring.

## How It Works

These methods provide audio-specific operations through the facade pattern.
Configure any model that implements `IAudioEventDetector` via
`IFullModel{` and then use
these methods for audio event detection.

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectAudioEvents(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Detects audio events in the given audio using the configured model's default threshold. |
| `DetectAudioEvents(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,)` | Detects audio events with a custom confidence threshold. |
| `DetectAudioEventsAsync(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,CancellationToken)` | Detects audio events asynchronously without blocking the calling thread. |
| `DetectSpecificAudioEvents(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,IReadOnlyList<String>)` | Detects only specific event types, filtering out everything else. |
| `DetectSpecificAudioEvents(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>,IReadOnlyList<String>,)` | Detects specific event types with a custom confidence threshold. |
| `GetAudioEventDetector(AiModelBuilder<,Tensor<>,Tensor<>>)` | Extracts and casts the configured model to `IAudioEventDetector`. |
| `GetAudioEventProbabilities(AiModelBuilder<,Tensor<>,Tensor<>>,Tensor<>)` | Gets raw frame-level event probabilities for all classes without thresholding. |
| `GetAudioEventTimeResolution(AiModelBuilder<,Tensor<>,Tensor<>>)` | Gets the time resolution of the configured audio event detection model. |
| `GetSupportedAudioEvents(AiModelBuilder<,Tensor<>,Tensor<>>)` | Gets the list of event types the configured audio model can detect. |
| `StartAudioEventStreaming(AiModelBuilder<,Tensor<>,Tensor<>>)` | Starts a streaming event detection session for real-time audio monitoring. |
| `StartAudioEventStreaming(AiModelBuilder<,Tensor<>,Tensor<>>,Int32,)` | Starts a streaming event detection session with custom settings. |

