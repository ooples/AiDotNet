---
title: "BandSplitRNNEnhancerOptions"
description: "Configuration options for the Band-Split RNN enhancement model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Enhancement`

Configuration options for the Band-Split RNN enhancement model.

## For Beginners

Imagine you're in a noisy room trying to hear someone speak.
Band-Split RNN works like having multiple specialized listeners, each focused on a
different pitch range (bass, midrange, treble). Each listener cleans up their range,
and then they combine their results. This divide-and-conquer approach works very well
because different types of noise affect different frequency ranges.

## How It Works

Band-Split RNN (Luo & Yu, 2023) splits the spectrogram into non-overlapping frequency
bands, processes each band with a shared RNN, then fuses the bands. Originally designed
for music source separation, it also excels at speech enhancement by treating noise as
a "source" to separate. Band-Split RNN achieves state-of-the-art on both music separation
and speech enhancement benchmarks.

## Properties

| Property | Summary |
|:-----|:--------|
| `BandRnnHiddenSize` | Gets or sets the RNN hidden size per band. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FFTSize` | Gets or sets the FFT size. |
| `FusionDim` | Gets or sets the band fusion hidden dimension. |
| `HopLength` | Gets or sets the hop length. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBands` | Gets or sets the number of frequency bands. |
| `NumFreqBins` | Gets or sets the number of frequency bins. |
| `NumRnnLayers` | Gets or sets the number of RNN layers per band. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("small", "medium", or "large"). |

