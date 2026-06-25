---
title: "BandSplitRNNOptions"
description: "Configuration options for the BandSplitRNN source separation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SourceSeparation`

Configuration options for the BandSplitRNN source separation model.

## For Beginners

BandSplitRNN works like a team of specialists: each specialist listens to a
specific frequency range (e.g., bass frequencies, mid-range, treble), learns to identify what
belongs to each instrument in their range, and then they all compare notes to produce a consistent
separation across the full frequency spectrum. This "divide and share" approach works better than
trying to process all frequencies at once.

## How It Works

BandSplitRNN (Luo and Yu, 2023) is the original Band-Split RNN model designed specifically for
music source separation. It splits the spectrogram into non-overlapping frequency bands, processes
each with a shared band-level RNN, applies cross-band fusion via a sequence-level RNN, and then
reconstructs source-specific masks. It achieves 10.0+ dB SDR on MUSDB18-HQ.

## Properties

| Property | Summary |
|:-----|:--------|
| `BandRnnHiddenSize` | Gets or sets the hidden size of the band-level RNN. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FftSize` | Gets or sets the FFT window size. |
| `FusionDim` | Gets or sets the band fusion hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBandRnnLayers` | Gets or sets the number of band-level RNN layers. |
| `NumBands` | Gets or sets the number of frequency bands to split into. |
| `NumFreqBins` | Gets or sets the number of frequency bins. |
| `NumSequenceRnnLayers` | Gets or sets the number of sequence-level RNN layers. |
| `NumStems` | Gets or sets the number of stems/sources. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SequenceRnnHiddenSize` | Gets or sets the hidden size of the sequence-level (cross-band) RNN. |
| `Sources` | Gets or sets the source names to separate. |
| `WeightDecay` | Gets or sets the weight decay. |

