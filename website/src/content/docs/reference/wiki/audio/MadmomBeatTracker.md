---
title: "MadmomBeatTracker<T>"
description: "Madmom-style neural beat tracker using bidirectional RNNs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Madmom-style neural beat tracker using bidirectional RNNs.

## For Beginners

This model listens to music and finds exactly where each beat falls,
like a musician tapping their foot in time. It can tell you the tempo (beats per minute) and
mark every beat position, which is essential for music synchronization, DJ software, and
automatic remixing.

**Usage:**

## How It Works

The Madmom beat tracking system (Bock et al., 2016) uses a recurrent neural network to detect
beat positions and downbeat positions in audio. It combines spectrogram features with bidirectional
RNNs and a dynamic Bayesian network for beat tracking, achieving state-of-the-art results on
multiple beat tracking benchmarks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MadmomBeatTracker(NeuralNetworkArchitecture<>,MadmomBeatTrackerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Madmom Beat Tracker in native training mode. |
| `MadmomBeatTracker(NeuralNetworkArchitecture<>,String,MadmomBeatTrackerOptions)` | Creates a Madmom Beat Tracker in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxBPM` |  |
| `MinBPM` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOnsetStrength(Tensor<>)` |  |
| `DetectDownbeats(Tensor<>,BeatTrackingResult<>)` |  |
| `EstimateTempo(Tensor<>)` |  |
| `GetTempoHypotheses(Tensor<>,Int32)` |  |
| `Track(Tensor<>)` |  |
| `TrackAsync(Tensor<>,CancellationToken)` |  |

