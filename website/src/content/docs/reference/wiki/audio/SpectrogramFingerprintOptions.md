---
title: "SpectrogramFingerprintOptions"
description: "Configuration options for spectrogram fingerprinting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Fingerprinting`

Configuration options for spectrogram fingerprinting.

## For Beginners

These options configure the SpectrogramFingerprint model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectrogramFingerprintOptions` | Initializes a new instance with default values. |
| `SpectrogramFingerprintOptions(SpectrogramFingerprintOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FftSize` | Gets or sets the FFT size. |
| `HopLength` | Gets or sets the hop length. |
| `MaxPeaksPerWindow` | Gets or sets the maximum peaks per analysis window. |
| `PeakNeighborhood` | Gets or sets the peak detection neighborhood size. |
| `PeakThreshold` | Gets or sets the minimum peak magnitude threshold. |
| `PeakWindowSizeFrames` | Gets or sets the window size in frames for peak selection. |
| `SampleRate` | Gets or sets the sample rate. |
| `TargetZoneEnd` | Gets or sets the target zone end (frames ahead). |
| `TargetZoneStart` | Gets or sets the target zone start (frames ahead). |

