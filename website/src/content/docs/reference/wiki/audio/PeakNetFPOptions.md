---
title: "PeakNetFPOptions"
description: "Configuration options for the PeakNetFP spectral peak-based fingerprinting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Fingerprinting`

Configuration options for the PeakNetFP spectral peak-based fingerprinting model.

## For Beginners

PeakNetFP identifies songs by finding the "peaks" in their sound
spectrum (like the loudest frequencies at each moment) and then uses AI to turn those peaks
into a compact code. It's a hybrid approach that combines classical Shazam-like peak picking
with modern neural network encoding.

## How It Works

PeakNetFP combines traditional spectral peak picking with a neural network for robust audio
fingerprinting. It detects spectral peaks in the spectrogram and uses a CNN to encode peak
constellations into compact binary hashes, offering both speed and robustness.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseFilters` | Gets or sets the base filter count. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the fingerprint embedding dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `MatchThreshold` | Gets or sets the cosine similarity threshold for considering a fingerprint match. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumEncoderBlocks` | Gets or sets the number of encoder blocks. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PeakNeighborhood` | Gets or sets the peak neighborhood size for non-max suppression. |
| `PeaksPerFrame` | Gets or sets the number of spectral peaks to select per frame. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SegmentDurationSec` | Gets or sets the segment duration in seconds. |
| `Temperature` | Gets or sets the contrastive loss temperature. |

