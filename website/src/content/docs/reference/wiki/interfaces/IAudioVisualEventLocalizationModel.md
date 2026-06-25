---
title: "IAudioVisualEventLocalizationModel<T>"
description: "Defines the contract for audio-visual event localization models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for audio-visual event localization models.

## For Beginners

Finding events in videos using sight AND sound!

Key capabilities:

- Temporal localization: When does the dog bark? (2.3s - 4.1s)
- Spatial localization: Where is the barking dog? (bounding box)
- Event classification: What kind of event is it? (animal sound)
- Multi-event detection: Find all events in a video

Use cases:

- Video surveillance: Detect glass breaking sounds and locate the window
- Sports analysis: Find and timestamp all goals using crowd cheering
- Content moderation: Detect and locate inappropriate audio-visual content

## How It Works

Audio-visual event localization identifies WHEN and WHERE events occur
in video by jointly analyzing audio and visual streams. This goes beyond
simple detection to provide precise temporal boundaries and spatial locations.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedEventCategories` | Gets the supported event categories. |
| `TemporalResolution` | Gets the temporal resolution in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnswerEventQuestion(Tensor<>,IEnumerable<Tensor<>>,String,Double)` | Answers questions about events in the video. |
| `ClassifyEvent(Tensor<>,IEnumerable<Tensor<>>,IEnumerable<String>)` | Classifies a pre-segmented event. |
| `ComputeEventAttention(Tensor<>,IEnumerable<Tensor<>>)` | Computes event-level audio-visual attention. |
| `DetectAnomalies(Tensor<>,IEnumerable<Tensor<>>,Double)` | Detects anomalous events that don't match expected patterns. |
| `DetectEvents(Tensor<>,IEnumerable<Tensor<>>,Double)` | Detects and localizes all audio-visual events in a video. |
| `DetectSpecificEvents(Tensor<>,IEnumerable<Tensor<>>,IEnumerable<String>,Double)` | Detects events of specific categories. |
| `DetectSyncEvents(Tensor<>,IEnumerable<Tensor<>>,Double)` | Detects audio-visual synchronization events (e.g., lip sync). |
| `GenerateDenseCaptions(Tensor<>,IEnumerable<Tensor<>>,Double)` | Generates dense event captions for the entire video. |
| `GenerateProposals(Tensor<>,IEnumerable<Tensor<>>,Double)` | Generates temporal proposals for potential events. |
| `LocalizeEventByDescription(Tensor<>,IEnumerable<Tensor<>>,String,Double)` | Localizes a specific event described in text. |
| `SegmentScenes(Tensor<>,IEnumerable<Tensor<>>,Double)` | Segments video into coherent audio-visual scenes. |
| `TrackEvent(Tensor<>,IEnumerable<Tensor<>>,AudioVisualEvent,Double)` | Tracks an event across time. |

