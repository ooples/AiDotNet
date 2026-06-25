---
title: "CREPE<T>"
description: "CREPE (Convolutional Representation for Pitch Estimation) neural pitch detector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

CREPE (Convolutional Representation for Pitch Estimation) neural pitch detector.

## For Beginners

CREPE listens to a single voice or instrument and tells you exactly
what note is being played. It's like a very accurate guitar tuner powered by AI. It takes
small chunks of audio and outputs which pitch (frequency) is most likely present.

**Usage:**

## How It Works

CREPE (Kim et al., 2018) is a deep convolutional network for monophonic pitch detection.
It operates directly on the audio waveform (1024 samples at 16 kHz) and outputs a
360-dimensional vector representing pitch salience across 20-cent bins from C1 to B7.
CREPE outperforms traditional methods (YIN, pYIN) especially in noisy conditions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CREPE(NeuralNetworkArchitecture<>,CREPEOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a CREPE model in native training mode. |
| `CREPE(NeuralNetworkArchitecture<>,String,CREPEOptions)` | Creates a CREPE model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxPitch` |  |
| `MinPitch` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectPitch(Tensor<>)` |  |
| `DetectPitchWithConfidence(Tensor<>)` |  |
| `ExtractDetailedPitchContour(Tensor<>,Int32)` |  |
| `ExtractPitchContour(Tensor<>,Int32)` |  |
| `GetCentsDeviation()` |  |
| `MidiToPitch(Double)` |  |
| `PitchToMidi()` |  |
| `PitchToNoteName()` |  |

