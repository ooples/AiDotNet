---
title: "MelodyExtractor<T>"
description: "Neural Melody Extractor that identifies the primary melodic line from polyphonic audio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Neural Melody Extractor that identifies the primary melodic line from polyphonic audio.

## For Beginners

When you listen to a song, you can usually hum along to the main
melody even though many instruments are playing. This model does the same thing - it finds
and extracts just the main tune from a full song, ignoring background harmonies and rhythms.

**Usage:**

## How It Works

The Melody Extractor identifies the primary melodic line from a polyphonic audio recording
using a neural network. Unlike pitch detection (which finds any pitch), melody extraction
specifically tracks the dominant melody even when other instruments are playing simultaneously.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MelodyExtractor(NeuralNetworkArchitecture<>,MelodyExtractorOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Melody Extractor in native training mode. |
| `MelodyExtractor(NeuralNetworkArchitecture<>,String,MelodyExtractorOptions)` | Creates a Melody Extractor in ONNX inference mode. |

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

