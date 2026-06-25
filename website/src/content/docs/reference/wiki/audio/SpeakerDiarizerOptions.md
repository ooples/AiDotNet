---
title: "SpeakerDiarizerOptions"
description: "Configuration options for speaker diarization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for speaker diarization.

## For Beginners

These options configure the SpeakerDiarizer model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeakerDiarizerOptions` | Initializes a new instance with default values. |
| `SpeakerDiarizerOptions(SpeakerDiarizerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClusteringThreshold` | Gets or sets the clustering threshold (cosine similarity). |
| `EmbeddingDimension` | Gets or sets the embedding dimension. |
| `EmbeddingModelPath` | Gets or sets the path to the embedding model. |
| `HopDurationSeconds` | Gets or sets the hop duration in seconds. |
| `MaxSpeakers` | Gets or sets the maximum number of speakers (null for auto). |
| `MinTurnDuration` | Gets or sets the minimum turn duration in seconds. |
| `SampleRate` | Gets or sets the sample rate. |
| `WindowDurationSeconds` | Gets or sets the window duration in seconds. |

