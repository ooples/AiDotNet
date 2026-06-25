---
title: "PitchShift<T>"
description: "Shifts the pitch of audio without changing tempo using WSOLA (Waveform Similarity Overlap-Add)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Audio`

Shifts the pitch of audio without changing tempo using WSOLA (Waveform Similarity Overlap-Add).

## For Beginners

Pitch shifting makes audio sound higher or lower,
like the difference between a high and low voice, while keeping the same duration.

## How It Works

**Algorithm:** Uses WSOLA (Waveform Similarity Overlap-Add) for time-stretching
followed by resampling. This preserves audio quality better than simple resampling.

**When to use:**

- Speech recognition to handle different voice pitches
- Music analysis to handle different keys
- Voice cloning and synthesis training

**Semitone reference:** 12 semitones = 1 octave (doubling/halving frequency)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PitchShift(Double,Double,Double,Int32)` | Creates a new pitch shift augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSemitones` | Gets the maximum pitch shift in semitones. |
| `MinSemitones` | Gets the minimum pitch shift in semitones. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Tensor<>,AugmentationContext<>)` |  |
| `ApplyPitchShift(Tensor<>,Double)` | Applies pitch shift using WSOLA time-stretching followed by resampling. |
| `FindBestOffset(Double[],Double[],Int32,Int32,Int32,Int32)` | Finds the best offset for frame alignment using cross-correlation. |
| `GetParameters` |  |
| `ResampleLinear(Double[],Int32)` | Resamples audio to target length using linear interpolation. |
| `TimeStretchWSOLA(Double[],Double)` | Time-stretches audio using WSOLA algorithm. |

