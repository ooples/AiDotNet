---
title: "YinPitchDetector<T>"
description: "YIN pitch detection algorithm implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Pitch`

YIN pitch detection algorithm implementation.

## For Beginners

YIN finds pitch by looking at how similar a signal is to itself!

The key insight:

- A periodic (pitched) signal repeats itself
- If we shift the signal and compare it to the original, it should match at the period

How YIN works:

1. Difference function: For each possible lag, calculate how different the signal

is from a shifted version of itself

2. Cumulative mean normalization: Normalize the difference to handle varying energy
3. Absolute threshold: Find the first lag where normalized difference is below threshold
4. Parabolic interpolation: Refine the estimate for sub-sample accuracy

Why YIN is popular:

- More accurate than simple autocorrelation
- Fewer octave errors (detecting 2x or 0.5x the actual pitch)
- Works well for speech and musical instruments
- Relatively fast computation

Parameters:

- Threshold: How much similarity is required (lower = stricter)
- Frame size: Longer frames = lower min pitch, more latency

## How It Works

YIN is a widely-used pitch detection algorithm known for its accuracy and
relatively low computational cost. It was developed by de Cheveigné and Kawahara in 2002.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YinPitchDetector(Int32,Double,Double,Double,Int32)` | Creates a YIN pitch detector with default parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCmndf(Double[])` | Computes the cumulative mean normalized difference function. |
| `ComputeDifference(Double[],Int32)` | Computes the difference function d(tau). |
| `DetectPitchInternal(Double[])` |  |
| `FindBestTau(Double[],Int32,Int32)` | Finds the best tau value using absolute threshold method. |
| `ParabolicInterpolation(Double[],Int32)` | Applies parabolic interpolation for sub-sample accuracy. |

