---
title: "GraFPrintOptions"
description: "Configuration options for the GraFPrint graph-based audio fingerprinting model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Fingerprinting`

Configuration options for the GraFPrint graph-based audio fingerprinting model.

## For Beginners

GraFPrint treats a song's spectrogram as a network (graph) of connected
sound points, then uses a special AI called a graph neural network to turn that network into
a fingerprint. This approach captures how different parts of the sound relate to each other,
making it very robust to distortions like noise or tempo changes.

## How It Works

GraFPrint uses graph neural networks to model spectro-temporal relationships in audio for
robust fingerprinting. It constructs a graph from spectrogram features where nodes represent
time-frequency points and edges capture local relationships, then applies GNN layers to
produce compact fingerprint embeddings.

## Properties

| Property | Summary |
|:-----|:--------|
| `DisableFusedOptimizerStep` | Disables the fused-Adam optimizer step inside the compiled training plan, falling back to eager Adam for the parameter update only. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the fingerprint embedding dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `GnnHiddenDim` | Gets or sets the GNN hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `KNeighbors` | Gets or sets the k-nearest neighbors for graph construction. |
| `LRSchedulerTMax` | Maximum step count for the cosine annealing LR scheduler. |
| `LearningRate` | AdamW learning rate. |
| `MatchThreshold` | Gets or sets the cosine similarity threshold for considering a fingerprint match. |
| `MaxGradNorm` | Maximum global L2 norm for gradients per training step. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of graph attention heads. |
| `NumGnnLayers` | Gets or sets the number of GNN layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SegmentDurationSec` | Gets or sets the segment duration in seconds. |
| `Temperature` | Gets or sets the contrastive loss temperature. |

