---
title: "AudioEventDetector<T>"
description: "Audio event detection model for identifying sounds in audio (AudioSet-style)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

Audio event detection model for identifying sounds in audio (AudioSet-style).

## For Beginners

Audio event detection answers "What sounds are in this audio?":

- Human sounds: speech, laughter, coughing, footsteps
- Animal sounds: dog barking, bird singing, cat meowing
- Music: instruments, genres, singing
- Environmental: traffic, rain, wind, construction

Usage with ONNX:

Usage with training:

## How It Works

Detects various audio events like speech, music, environmental sounds, and more.
Based on AudioSet ontology with 527+ event classes organized hierarchically.

**Architecture:** This model extends `AudioClassifierBase`
and implements `IAudioEventDetector` for multi-label event detection.
Unlike single-label classification, event detection identifies multiple overlapping
events with their temporal boundaries.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioEventDetector(AudioEventDetectorOptions)` | Creates an AudioEventDetector with legacy options only (native mode). |
| `AudioEventDetector(NeuralNetworkArchitecture<>,AudioEventDetectorOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an AudioEventDetector for native training mode. |
| `AudioEventDetector(NeuralNetworkArchitecture<>,String,AudioEventDetectorOptions)` | Creates an AudioEventDetector for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` | Gets the event labels (alias for SupportedEvents for legacy API compatibility). |
| `SupportedEvents` | Gets the list of event types this model can detect. |
| `TimeResolution` | Gets the time resolution for event detection in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClassifyWithRules(Tensor<>,Tensor<>)` | Rule-based fallback classification when neither ONNX nor native neural network mode is available. |
| `CreateAsync(AudioEventDetectorOptions,IProgress<Double>,CancellationToken)` | Creates an AudioEventDetector asynchronously with model download. |
| `CreateNewInstance` | Creates a new instance for deserialization. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Detect(Tensor<>)` | Detects audio events in the audio stream. |
| `Detect(Tensor<>,)` | Detects audio events with custom threshold. |
| `DetectAsync(Tensor<>,CancellationToken)` | Detects audio events asynchronously. |
| `DetectFrame(Tensor<>)` | Detects a single frame (no windowing) - legacy API. |
| `DetectLegacy(Tensor<>)` | Detects audio events in the given audio (legacy API). |
| `DetectLegacyAsync(Tensor<>,CancellationToken)` | Detects audio events asynchronously (legacy API). |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` | Detects specific events only. |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` | Detects specific events only with custom threshold. |
| `DetectTopK(Tensor<>,Int32)` | Gets the top K events for a single frame (legacy API). |
| `Dispose(Boolean)` | Disposes managed resources. |
| `GetEventProbabilities(Tensor<>)` | Gets frame-level event probabilities. |
| `GetModelMetadata` | Gets model metadata. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers. |
| `PostprocessOutput(Tensor<>)` | Post-processes model output. |
| `PredictCore(Tensor<>)` | Predicts output for the given input. |
| `PreprocessAudio(Tensor<>)` | Preprocesses audio for the model. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `StartStreamingSession` | Starts a streaming event detection session. |
| `StartStreamingSession(Int32,)` | Starts a streaming event detection session with custom settings. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single example. |
| `UpdateParameters(Vector<>)` | Updates network parameters from a flattened vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CommonEventLabels` | Common audio event categories from AudioSet. |

