---
title: "SpeakerDiarizer<T>"
description: "Performs speaker diarization (who spoke when) on audio recordings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

Performs speaker diarization (who spoke when) on audio recordings.

## For Beginners

Diarization is like automatically labeling a meeting
recording with "Speaker A: 0:00-0:15, Speaker B: 0:15-0:45..."

The process:

1. Split audio into short segments
2. Extract speaker embeddings for each segment
3. Cluster similar embeddings together
4. Each cluster represents a different speaker

Common applications:

- Meeting transcription
- Call center analytics
- Podcast processing

Usage:

## How It Works

Speaker diarization segments audio by speaker, answering "who spoke when?"
It uses embeddings from sliding windows and clustering to identify speaker turns.

This class supports both:

- **ONNX mode**: Load pre-trained models for fast inference
- **Native training mode**: Train from scratch using the layer architecture

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeakerDiarizer(NeuralNetworkArchitecture<>,SpeakerDiarizerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a new speaker diarizer in native training mode. |
| `SpeakerDiarizer(NeuralNetworkArchitecture<>,String,SpeakerDiarizerOptions)` | Creates a new speaker diarizer in ONNX inference mode. |
| `SpeakerDiarizer(SpeakerDiarizerOptions)` | Creates a new speaker diarizer with legacy options only. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusteringThreshold` | Gets the clustering threshold. |
| `IsOnnxMode` | Gets whether the model is operating in ONNX inference mode. |
| `MinSegmentDuration` | Gets the minimum segment duration in seconds. |
| `MinTurnDuration` | Gets the minimum turn duration in seconds. |
| `SampleRate` | Gets the sample rate. |
| `SupportsOverlapDetection` | Gets whether this model can detect overlapping speech. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Diarize(Tensor<>,Nullable<Int32>,Int32,Int32)` | Performs speaker diarization on audio. |
| `DiarizeAsync(Tensor<>,Nullable<Int32>,Int32,Int32,CancellationToken)` | Performs speaker diarization asynchronously. |
| `DiarizeLegacy(Tensor<>)` | Performs diarization on audio (legacy API). |
| `DiarizeLegacy(Vector<>)` | Performs diarization on audio (legacy API). |
| `DiarizeWithKnownSpeakers(Tensor<>,IReadOnlyList<SpeakerProfile<>>,Boolean)` | Performs diarization with known speaker profiles. |
| `Dispose` | Disposes resources. |
| `Dispose(Boolean)` | Disposes managed resources. |
| `ExtractSpeakerEmbeddings(Tensor<>,DiarizationResult<>)` | Gets speaker embeddings for each detected speaker. |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the neural network layers. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PredictCore(Tensor<>)` | Predicts output for the given input. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `RefineDiarization(Tensor<>,DiarizationResult<>,)` | Refines diarization result by re-segmenting with different parameters. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single example. |
| `UpdateParameters(Vector<>)` | Updates model parameters. |

