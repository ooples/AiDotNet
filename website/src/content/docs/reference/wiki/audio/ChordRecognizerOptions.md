---
title: "ChordRecognizerOptions"
description: "Configuration options for chord recognition."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for chord recognition.

## For Beginners

These options configure the ChordRecognizer model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChordRecognizerOptions` | Initializes a new instance with default values. |
| `ChordRecognizerOptions(ChordRecognizerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FftSize` | Gets or sets the FFT size. |
| `HopLength` | Gets or sets the hop length. |
| `MinChromaEnergy` | Gets or sets the minimum chroma energy to consider. |
| `MinConfidence` | Gets or sets the minimum confidence for chord detection. |
| `MinSegmentDuration` | Gets or sets the minimum segment duration in seconds. |
| `SampleRate` | Gets or sets the sample rate. |

