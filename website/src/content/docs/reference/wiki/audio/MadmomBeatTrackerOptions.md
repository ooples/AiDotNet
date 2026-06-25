---
title: "MadmomBeatTrackerOptions"
description: "Configuration options for the Madmom-style neural beat tracker."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the Madmom-style neural beat tracker.

## For Beginners

This model listens to music and finds exactly where each beat falls,
like a musician tapping their foot in time. It can tell you the tempo (beats per minute) and
mark every beat position, which is essential for music synchronization, DJ software, and
automatic remixing.

## How It Works

The Madmom beat tracking system (Bock et al., 2016) uses a recurrent neural network to detect
beat positions and downbeat positions in audio. It combines spectrogram features with bidirectional
RNNs and a dynamic Bayesian network for beat tracking, achieving state-of-the-art results on
multiple beat tracking benchmarks.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `MinBeatInterval` | Gets or sets the minimum inter-beat interval in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBands` | Gets or sets the number of spectrogram bands. |
| `NumRnnLayers` | Gets or sets the number of RNN layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PeakThreshold` | Gets or sets the peak picking threshold for beat detection. |
| `RnnHiddenSize` | Gets or sets the RNN hidden size. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |

