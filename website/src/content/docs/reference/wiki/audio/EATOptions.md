---
title: "EATOptions"
description: "Configuration options for the EAT (Efficient Audio Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the EAT (Efficient Audio Transformer) model.

## For Beginners

EAT is designed to be more efficient while maintaining accuracy.
It uses a teacher-student framework where a smaller student model learns from a larger
teacher, making training much faster. Think of it as a student who learns efficiently
by watching an expert rather than figuring everything out alone.

## How It Works

EAT (Chen et al., 2024) is an efficient self-supervised audio pre-training model that
achieves competitive performance with significantly less compute than previous methods.
It reaches 49.7% mAP on AudioSet-2M using only 10% of the pre-training data and compute
of BEATs.

**References:**

- Paper: "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer" (Chen et al., 2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropoutRate` | Gets or sets the attention dropout rate. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmaDecay` | Gets or sets the EMA (Exponential Moving Average) decay for teacher model updates. |
| `EmbeddingDim` | Gets or sets the embedding dimension. |
| `FMax` | Gets or sets the maximum frequency for mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for mel filterbank. |
| `FeedForwardDim` | Gets or sets the feed-forward network dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between FFT frames. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `MaskRatio` | Gets or sets the mask ratio for self-supervised pre-training. |
| `MinMaskSpanLength` | Gets or sets the minimum span length for masking. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumEncoderLayers` | Gets or sets the number of Transformer encoder layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PatchSize` | Gets or sets the patch size. |
| `PatchStride` | Gets or sets the patch stride. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `WarmUpSteps` | Gets or sets the number of warm-up steps. |
| `WindowOverlap` | Gets or sets the window overlap ratio. |
| `WindowSize` | Gets or sets the window size in seconds. |

