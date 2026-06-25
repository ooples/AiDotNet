---
title: "TitaNetOptions"
description: "Configuration options for the TitaNet speaker verification and embedding model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for the TitaNet speaker verification and embedding model.

## For Beginners

TitaNet is NVIDIA's advanced voice fingerprinting model. It uses
efficient convolutions to process speech and creates a compact embedding that uniquely
identifies a speaker. It comes in three sizes: Small (S), Medium (M), and Large (L).

## How It Works

TitaNet (Koluguri et al., ICASSP 2022) is NVIDIA's speaker embedding model based on
1D depth-wise separable convolutions with Squeeze-Excitation and global context. TitaNet-L
achieves 0.68% EER on VoxCeleb1-O, outperforming ECAPA-TDNN.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentivePoolingDim` | Gets or sets the attentive statistics pooling hidden dimension. |
| `ConvKernelSize` | Gets or sets the depth-wise separable convolution kernel size. |
| `DefaultThreshold` | Gets or sets the default cosine similarity threshold for verification. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the output embedding dimension. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `FftSize` | Gets or sets the FFT window size in samples. |
| `HopLength` | Gets or sets the hop length between frames in samples. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `MinDurationSeconds` | Gets or sets the minimum audio duration in seconds for reliable embedding. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumEncoderBlocks` | Gets or sets the number of encoder blocks (prolog + body + epilog). |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SEReductionRatio` | Gets or sets the SE (Squeeze-Excitation) reduction ratio. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant (S, M, or L). |
| `WeightDecay` | Gets or sets the weight decay for regularization. |

