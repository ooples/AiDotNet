---
title: "NeuralFPOptions"
description: "Configuration options for the Neural Audio Fingerprint (NeuralFP) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Fingerprinting`

Configuration options for the Neural Audio Fingerprint (NeuralFP) model.

## For Beginners

NeuralFP creates "audio IDs" using AI instead of traditional signal
processing. It converts a short audio clip into a small vector of numbers (like a barcode).
Two recordings of the same song will produce very similar vectors, even if one is noisy
or compressed. This enables Shazam-like audio identification.

## How It Works

NeuralFP (Chang et al., 2021) uses a neural network to learn compact audio fingerprints
for large-scale audio retrieval. It generates fixed-length embeddings from mel spectrograms
that are robust to noise, compression, and time-stretching. The model uses contrastive
learning to ensure similar audio produces similar fingerprints.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseFilters` | Gets or sets the base filter count for convolutional layers. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the fingerprint embedding dimension. |
| `FftSize` | Gets or sets the FFT window size in samples. |
| `HopLength` | Gets or sets the hop length between frames in samples. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `MatchThreshold` | Gets or sets the cosine similarity threshold for considering a fingerprint match. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumConvBlocks` | Gets or sets the number of convolutional blocks in the encoder. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SegmentDurationSec` | Gets or sets the segment duration in seconds for fingerprint extraction. |
| `Temperature` | Gets or sets the temperature for contrastive loss during training. |

