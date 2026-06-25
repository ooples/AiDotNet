---
title: "WavLMOptions"
description: "Configuration options for the WavLM self-supervised speech model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Foundations`

Configuration options for the WavLM self-supervised speech model.

## For Beginners

WavLM is an improved version of HuBERT that's especially good at
understanding noisy speech and telling different speakers apart. It was trained to
understand speech even with background noise, making it more robust in real-world
conditions.

## How It Works

WavLM (Chen et al., 2022, Microsoft) extends HuBERT with gated relative position bias
and denoising pre-training. It achieves state-of-the-art on the SUPERB benchmark across
speech recognition, speaker verification, speaker diarization, and more.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FeatureEncoderDim` | Gets or sets the CNN feature encoder output dimension. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `HiddenDim` | Gets or sets the transformer hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `UseGatedRelativePositionBias` | Gets or sets whether to use gated relative position bias. |
| `Variant` | Gets or sets the model variant ("base", "base+", or "large"). |

