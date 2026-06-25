---
title: "WavLMSpeakerOptions"
description: "Configuration options for the WavLM Speaker verification and embedding model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Speaker`

Configuration options for the WavLM Speaker verification and embedding model.

## For Beginners

WavLM was originally trained to understand speech in general (like a
language student who listens to lots of conversations). When specialized for speaker verification,
it becomes excellent at recognizing individual voices because it already understands the deep
structure of speech. It works especially well in noisy environments.

## How It Works

WavLM (Chen et al., 2022) is a self-supervised speech model that, when fine-tuned for speaker
verification, achieves state-of-the-art results with 0.59% EER on VoxCeleb1 test set. It uses
a Transformer encoder pre-trained with masked speech prediction and denoising objectives,
making it robust to noisy conditions.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultThreshold` | Gets or sets the default cosine similarity threshold. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the output speaker embedding dimension. |
| `FeatureEncoderDim` | Gets or sets the feature encoder dimension (CNN output). |
| `FeedForwardDim` | Gets or sets the feed-forward hidden dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the Transformer hidden dimension. |
| `HopLength` | Gets or sets the hop length between frames. |
| `LearningRate` | Gets or sets the learning rate. |
| `MinDurationSeconds` | Gets or sets the minimum audio duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of Transformer layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PoolingStrategy` | Gets or sets the pooling strategy ("mean", "stats", or "attentive"). |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("base" or "large"). |
| `WeightDecay` | Gets or sets the weight decay. |

