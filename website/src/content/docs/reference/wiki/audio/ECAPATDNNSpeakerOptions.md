---
title: "ECAPATDNNSpeakerOptions"
description: "Configuration options for the ECAPA-TDNN speaker verification and embedding model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for the ECAPA-TDNN speaker verification and embedding model.

## For Beginners

ECAPA-TDNN creates a "voiceprint" for any speaker. It processes
audio through special layers that focus on the most important voice characteristics.
The result is a compact vector (embedding) that uniquely identifies a speaker's voice.

## How It Works

ECAPA-TDNN (Desplanques et al., Interspeech 2020) is a state-of-the-art speaker embedding
model that extends x-vector architecture with Squeeze-Excitation blocks, multi-layer feature
aggregation, and channel- and context-dependent statistics pooling. Achieves 0.87% EER on
VoxCeleb1 test set.

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets or sets the channel dimensions for each TDNN block. |
| `DefaultThreshold` | Gets or sets the default cosine similarity threshold for verification. |
| `Dilations` | Gets or sets the dilation factors for each TDNN block. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the output embedding dimension. |
| `FftSize` | Gets or sets the FFT window size in samples. |
| `HopLength` | Gets or sets the hop length between frames in samples. |
| `KernelSizes` | Gets or sets the TDNN kernel sizes (dilation factors). |
| `LearningRate` | Gets or sets the learning rate for training. |
| `MinDurationSeconds` | Gets or sets the minimum audio duration in seconds for reliable embedding. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PoolingDim` | Gets or sets the pooling dimension before the final embedding projection. |
| `Res2NetScale` | Gets or sets the Res2Net scale factor. |
| `SEBottleneckDim` | Gets or sets the SE (Squeeze-Excitation) bottleneck dimension. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

