---
title: "TempogramOptions"
description: "Configuration options for the neural Tempogram tempo estimation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the neural Tempogram tempo estimation model.

## For Beginners

A tempogram shows how the tempo (speed) of music changes over time.
This model creates a detailed map of tempo, which is useful for analyzing songs with
tempo changes, rubato (expressive timing), or live performances where the tempo varies.

## How It Works

The Tempogram model computes a tempo representation over time using a neural approach
to onset detection and autocorrelation-based tempo estimation. It provides both global
tempo and tempo curves for music with changing tempos.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxBPM` | Gets or sets the maximum BPM to detect. |
| `MinBPM` | Gets or sets the minimum BPM to detect. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumOnsetLayers` | Gets or sets the number of onset detector layers. |
| `NumTempoBins` | Gets or sets the number of tempo bins in the output. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `OnsetHiddenDim` | Gets or sets the hidden dimension of the onset detector. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `TempoWindowFrames` | Gets or sets the tempo estimation window in frames. |

