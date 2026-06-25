---
title: "SCNetOptions"
description: "Configuration options for the SCNet (Sparse Compression Network) source separation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SourceSeparation`

Configuration options for the SCNet (Sparse Compression Network) source separation model.

## For Beginners

SCNet is like a fast note-taker who summarizes a complex speech into
key points, processes just those points, and then expands them back to the full detail.
By compressing audio information before processing, it runs faster than methods that
process every frequency individually, while still producing high-quality separations.

## How It Works

SCNet (Chen et al., 2024) uses a sparse compression approach that compresses frequency features
into compact representations before processing with attention layers. This reduces computation
while maintaining separation quality. It achieves competitive results on MUSDB18-HQ with
significantly fewer parameters than Transformer-based models.

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressionDim` | Gets or sets the compression embedding dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FeedForwardDim` | Gets or sets the feed-forward expansion dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the Adam learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumClusters` | Gets or sets the number of compression clusters. |
| `NumDecoderBlocks` | Gets or sets the number of decoder blocks. |
| `NumEncoderBlocks` | Gets or sets the number of encoder blocks. |
| `NumFreqBins` | Gets or sets the number of frequency bins. |
| `NumStems` | Gets or sets the number of stems/sources. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Sources` | Gets or sets the source names to separate. |
| `WeightDecay` | Gets or sets the optional weight decay for custom optimizers. |

