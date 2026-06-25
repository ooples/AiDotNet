---
title: "IAudioEnhancer<T>"
description: "Defines the contract for audio enhancement models that improve audio quality."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for audio enhancement models that improve audio quality.

## For Beginners

Audio enhancement is like photo editing for sound!

Common use cases:

- Cleaning up podcast recordings (removing AC hum, keyboard clicks)
- Improving phone call quality (reducing background noise)
- Restoring old recordings (removing tape hiss, crackle)
- Video conferencing (echo cancellation, noise suppression)
- Hearing aids (speech enhancement in noisy environments)

How it works (simplified):

1. Analyze the audio to identify "noise" vs "signal"
2. Create a filter that reduces noise while keeping the signal
3. Apply the filter to produce cleaner audio

Modern approaches use neural networks that learn what clean audio
should sound like, producing much better results than traditional methods.

## How It Works

Audio enhancement encompasses various techniques to improve audio quality:

- Noise Reduction: Remove background noise while preserving speech/music
- Speech Enhancement: Improve speech intelligibility and quality
- Dereverberation: Remove room echo and reverb artifacts
- Echo Cancellation: Remove acoustic echo in communication systems
- Bandwidth Extension: Extend frequency range of narrowband audio

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` | Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum). |
| `LatencySamples` | Gets the processing latency in samples. |
| `NumChannels` | Gets the number of audio channels supported. |
| `SampleRate` | Gets the sample rate this enhancer operates at. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Enhance(Tensor<>)` | Enhances audio quality by reducing noise and artifacts. |
| `EnhanceWithReference(Tensor<>,Tensor<>)` | Enhances audio with a reference signal for echo cancellation. |
| `EstimateNoiseProfile(Tensor<>)` | Estimates the noise profile from a segment of audio. |
| `ProcessChunk(Tensor<>)` | Processes audio in real-time streaming mode. |
| `ResetState` | Resets internal state for streaming mode. |

