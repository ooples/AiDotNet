---
title: "EnergyBasedVad<T>"
description: "Simple energy-based voice activity detector (algorithmic, no neural network)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.VoiceActivity`

Simple energy-based voice activity detector (algorithmic, no neural network).

## For Beginners

This is the simplest type of VAD:

Basic idea: Speech is louder than silence!

- Compute the "energy" (sum of squared samples) for each frame
- If energy exceeds a threshold, it's probably speech

Enhanced features used here:

1. Energy: How loud is the signal?
2. Zero-crossings: How often does the signal cross zero?
- Speech: Medium zero-crossings (voiced sounds)
- Noise: High zero-crossings (random noise)
3. Spectral flatness: Is it tonal or noisy?
- Speech: Low flatness (has harmonic structure)
- Noise: High flatness (random spectrum)

Pros:

- Very fast (no neural network)
- Low latency
- Works well in quiet environments

Cons:

- Struggles with background noise
- May trigger on loud non-speech sounds
- Requires threshold tuning for different environments

For better noise robustness, use neural network-based VAD like SileroVad.

## How It Works

This is a basic VAD that detects speech based on signal energy (loudness).
It combines multiple features for more robust detection:

- Short-time energy
- Zero-crossing rate
- Spectral flatness

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnergyBasedVad(Int32,Int32,Double,Double,Double,Double,Boolean,Int32,Int32)` | Creates an energy-based VAD with default parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFrameProbability([])` |  |
| `ResetState` | Resets the VAD state including adaptive thresholds. |

