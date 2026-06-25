---
title: "MelodyExtractorOptions"
description: "Configuration options for the neural Melody Extraction model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the neural Melody Extraction model.

## For Beginners

When you listen to a song, you can usually hum along to the main
melody even though many instruments are playing. This model does the same thing—it finds
and extracts just the main tune from a full song, ignoring background harmonies and rhythms.

## How It Works

The Melody Extractor identifies the primary melodic line from a polyphonic audio recording
using a neural network. Unlike pitch detection (which finds any pitch), melody extraction
specifically tracks the dominant melody even when other instruments are playing simultaneously.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxFrequency` | Gets or sets the maximum frequency in Hz. |
| `MinFrequency` | Gets or sets the minimum frequency in Hz. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumPitchBins` | Gets or sets the number of pitch bins in the output. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `VoicingThreshold` | Gets or sets the voicing threshold. |

