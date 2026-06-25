---
title: "PANNsOptions"
description: "Configuration options for the PANNs (Pre-trained Audio Neural Networks) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the PANNs (Pre-trained Audio Neural Networks) model.

## For Beginners

PANNs is a family of CNN-based audio classifiers. Unlike Transformer
models (AST, BEATs), PANNs uses convolutional neural networks (CNNs) - the same technology
used for image recognition. CNNs are good at detecting local patterns (like specific
frequency shapes) and combining them into higher-level understanding (like "this is a dog bark").
PANNs is fast, well-tested, and widely used as a feature extractor for other audio tasks.

## How It Works

PANNs (Kong et al., IEEE/ACM TASLP 2020) provides a comprehensive set of pre-trained
CNN-based audio classification models. The flagship CNN14 model achieves 43.1% mAP on
AudioSet-2M and has become one of the most widely-used audio feature extractors.

**References:**

- Paper: "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition" (Kong et al., 2020)
- Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` | Gets or sets the base channel count. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the embedding dimension for the classification head. |
| `FMax` | Gets or sets the maximum frequency. |
| `FMin` | Gets or sets the minimum frequency. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the initial learning rate. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumBlocks` | Gets or sets the number of CNN blocks. |
| `NumMels` | Gets or sets the number of mel bands. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold. |
| `WarmUpSteps` | Gets or sets the warm-up steps. |
| `WindowOverlap` | Gets or sets the window overlap ratio. |
| `WindowSize` | Gets or sets the window size in seconds. |

