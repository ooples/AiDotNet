---
title: "SpectralSubtractionEnhancer<T>"
description: "Audio enhancer using spectral subtraction for noise reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

Audio enhancer using spectral subtraction for noise reduction.

## For Beginners

Think of it like this:

Imagine you're in a cafe trying to hear a friend:

- Noisy signal = friend's voice + cafe background noise
- If we know what the cafe sounds like alone (noise estimate)
- We can "subtract" the cafe sound to hear just the friend

Advantages:

- Simple and fast
- Low latency (good for real-time)
- Works well for stationary noise (AC hum, fan noise)

Limitations:

- Can introduce "musical noise" artifacts (twinkling sounds)
- Struggles with non-stationary noise (traffic, other speakers)
- May reduce speech quality if over-applied

This implementation includes:

- Adaptive noise estimation
- Spectral flooring (prevents negative magnitudes)
- Smoothing to reduce musical noise

## How It Works

Spectral subtraction is a classic noise reduction technique that:

1. Estimates the noise spectrum during silent periods
2. Subtracts the noise spectrum from the noisy signal spectrum
3. Reconstructs the cleaned signal

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralSubtractionEnhancer(Int32,Int32,Int32,Double,Double,Double,Boolean,Double)` | Initializes a new SpectralSubtractionEnhancer with default parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNoiseThreshold` | Gets the threshold for considering a frame as noise-only. |
| `ProcessSpectralFrame([],[])` |  |
| `ResetState` |  |
| `UpdateNoiseEstimate([])` | Updates the running noise estimate with new frame. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adaptiveNoiseEstimation` | Whether to use adaptive noise estimation. |
| `_alpha` | Over-subtraction factor (1.0 = exact subtraction, higher = more aggressive). |
| `_beta` | Spectral floor factor (prevents complete zeroing of bins). |
| `_previousMagnitudes` | Previous frame magnitudes for smoothing. |
| `_runningNoiseEstimate` | Running noise estimate. |
| `_smoothingFactor` | Smoothing factor for noise estimate updates. |

