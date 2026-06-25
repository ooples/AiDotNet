---
title: "Tempogram<T>"
description: "Neural Tempogram model for tempo estimation over time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Neural Tempogram model for tempo estimation over time.

## For Beginners

A tempogram shows how the tempo (speed) of music changes over time.
This model creates a detailed map of tempo, which is useful for analyzing songs with
tempo changes, rubato (expressive timing), or live performances where the tempo varies.

**Usage:**

## How It Works

The Tempogram model computes a tempo representation over time using a neural approach
to onset detection and autocorrelation-based tempo estimation. It provides both global
tempo and tempo curves for music with changing tempos.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Tempogram(NeuralNetworkArchitecture<>,String,TempogramOptions)` | Creates a Tempogram model in ONNX inference mode. |
| `Tempogram(NeuralNetworkArchitecture<>,TempogramOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Tempogram model in native training mode. |

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

