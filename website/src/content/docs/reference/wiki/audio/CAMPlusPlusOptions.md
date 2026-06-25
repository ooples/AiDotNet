---
title: "CAMPlusPlusOptions"
description: "Configuration options for the CAM++ (Context-Aware Masking Plus Plus) speaker model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for the CAM++ (Context-Aware Masking Plus Plus) speaker model.

## For Beginners

CAM++ is a lightweight speaker recognition model optimized for speed.
It uses a clever "context-aware masking" technique that helps it focus on the most important
parts of speech for identifying who is speaking, while ignoring silence and noise. This makes
it both fast and accurate—ideal for real-time applications.

## How It Works

CAM++ (Wang et al., 2023) is a fast and accurate speaker verification model that uses
context-aware masking with a densely connected time delay neural network (D-TDNN).
It processes variable-length utterances efficiently and achieves competitive EER results
while being significantly faster than Transformer-based approaches.

## Properties

| Property | Summary |
|:-----|:--------|
| `BottleneckDim` | Gets or sets the bottleneck dimension for D-TDNN blocks. |
| `DefaultThreshold` | Gets or sets the default cosine similarity threshold. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the output speaker embedding dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `GrowthRate` | Gets or sets the growth rate for dense connections. |
| `HopLength` | Gets or sets the hop length between frames. |
| `InitialChannels` | Gets or sets the initial channel dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaskingDim` | Gets or sets the context-aware masking dimension. |
| `MinDurationSeconds` | Gets or sets the minimum audio duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBlocks` | Gets or sets the number of D-TDNN blocks. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PoolingDim` | Gets or sets the pooling dimension before embedding projection. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `WeightDecay` | Gets or sets the weight decay. |

