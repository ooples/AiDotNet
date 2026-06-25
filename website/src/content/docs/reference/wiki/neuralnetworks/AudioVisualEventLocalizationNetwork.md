---
title: "AudioVisualEventLocalizationNetwork<T>"
description: "Neural network for audio-visual event localization - identifying WHEN and WHERE events occur in video by jointly analyzing audio and visual streams with precise temporal boundaries."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Neural network for audio-visual event localization - identifying WHEN and WHERE events occur
in video by jointly analyzing audio and visual streams with precise temporal boundaries.

## For Beginners

This model watches and listens to video simultaneously to find
specific events. For example, in a concert video it can identify:

- WHEN the guitar solo starts and ends (temporal localization)
- WHERE on screen the guitar player is (spatial localization)

It works by processing audio and video frames in parallel, then using cross-modal
attention to find moments where what's heard matches what's seen. This is useful for
video surveillance, sports analysis, and content moderation.

## How It Works

This network jointly analyzes audio and visual streams to identify when and where events
occur in video, producing precise temporal boundaries for detected events.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioVisualEventLocalizationNetwork` | Initializes a new instance with default architecture settings. |
| `AudioVisualEventLocalizationNetwork(NeuralNetworkArchitecture<>,Int32,Double,Int32,IEnumerable<String>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Nullable<Int32>,AudioVisualEventLocalizationOptions)` | Initializes a new instance of the AudioVisualEventLocalizationNetwork. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportedEventCategories` |  |
| `TemporalResolution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerEventQuestion(Tensor<>,IEnumerable<Tensor<>>,String,Double)` |  |
| `ClassifyEvent(Tensor<>,IEnumerable<Tensor<>>,IEnumerable<String>)` |  |
| `ComputeEventAttention(Tensor<>,IEnumerable<Tensor<>>)` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DetectAnomalies(Tensor<>,IEnumerable<Tensor<>>,Double)` |  |
| `DetectEvents(Tensor<>,IEnumerable<Tensor<>>,Double)` |  |
| `DetectSpecificEvents(Tensor<>,IEnumerable<Tensor<>>,IEnumerable<String>,Double)` |  |
| `DetectSyncEvents(Tensor<>,IEnumerable<Tensor<>>,Double)` |  |
| `GenerateDenseCaptions(Tensor<>,IEnumerable<Tensor<>>,Double)` |  |
| `GenerateProposals(Tensor<>,IEnumerable<Tensor<>>,Double)` |  |
| `GetDeterministicHashCode(String)` | Computes a deterministic hash code using the FNV-1a algorithm. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetParameters` |  |
| `InitializeLayers` |  |
| `LocalizeEventByDescription(Tensor<>,IEnumerable<Tensor<>>,String,Double)` |  |
| `PredictCore(Tensor<>)` |  |
| `SegmentScenes(Tensor<>,IEnumerable<Tensor<>>,Double)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetParameters(Vector<>)` |  |
| `TrackEvent(Tensor<>,IEnumerable<Tensor<>>,AudioVisualEvent,Double)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

