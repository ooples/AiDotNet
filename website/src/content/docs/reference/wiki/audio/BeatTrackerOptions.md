---
title: "BeatTrackerOptions"
description: "Configuration options for beat tracking."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for beat tracking.

## For Beginners

These options configure the BeatTracker model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BeatTrackerOptions` | Initializes a new instance with default values. |
| `BeatTrackerOptions(BeatTrackerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FftSize` | Gets or sets the FFT size for spectral analysis. |
| `HopLength` | Gets or sets the hop length for frame extraction. |
| `MaxTempo` | Gets or sets the maximum tempo to consider (BPM). |
| `MinTempo` | Gets or sets the minimum tempo to consider (BPM). |
| `SampleRate` | Gets or sets the sample rate of the audio. |
| `SmoothingWindow` | Gets or sets the smoothing window size for onset envelope. |
| `TempoFlexibility` | Gets or sets how flexible the tempo tracking is (0-1). |

