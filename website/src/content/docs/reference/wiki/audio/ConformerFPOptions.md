---
title: "ConformerFPOptions"
description: "Configuration options for the Conformer-based audio fingerprinting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Fingerprinting`

Configuration options for the Conformer-based audio fingerprinting model.

## For Beginners

ConformerFP uses a powerful AI architecture called "Conformer" that
combines two different ways of understanding audio: one that looks at the big picture (like
reading a whole sentence) and one that looks at local details (like reading letter by letter).
This combination makes it very good at creating fingerprints that can identify songs even
from noisy or distorted recordings.

## How It Works

ConformerFP applies the Conformer architecture (convolution-augmented Transformer) to audio
fingerprinting. It combines self-attention for global context with convolutions for local
feature extraction, producing highly robust fingerprints for large-scale audio retrieval.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvKernelSize` | Gets or sets the convolution kernel size for the Conformer conv module. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the fingerprint embedding dimension. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the Conformer hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `MatchThreshold` | Gets or sets the cosine similarity threshold for considering a fingerprint match. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of Conformer layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SegmentDurationSec` | Gets or sets the segment duration in seconds. |
| `Temperature` | Gets or sets the contrastive loss temperature. |

