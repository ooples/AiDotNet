---
title: "Data2Vec2Options"
description: "Configuration options for the data2vec 2.0 self-supervised audio foundation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Foundations`

Configuration options for the data2vec 2.0 self-supervised audio foundation model.

## For Beginners

data2vec 2.0 is a foundation model that learns by predicting its own
hidden representations - like studying by explaining things to yourself. Unlike HuBERT which
predicts discrete labels, data2vec predicts rich continuous features. Version 2.0 is much
faster to train while maintaining quality. It works for audio, images, and text.

## How It Works

data2vec 2.0 (Baevski et al., 2023, Meta) is a self-supervised learning framework that
predicts contextualized latent representations rather than modality-specific targets.
Version 2.0 is 16x faster than v1 through efficient data encoding and a novel self-distillation
objective. It achieves strong results on speech, vision, and language tasks with the same method.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvChannels` | Gets or sets the convolutional feature extractor channels. |
| `ConvKernels` | Gets or sets the convolutional feature extractor kernel sizes. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EMADecay` | Gets or sets the teacher EMA decay rate. |
| `FeedForwardDim` | Gets or sets the feed-forward inner dimension. |
| `HiddenDim` | Gets or sets the model hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaskProbability` | Gets or sets the masking probability for training. |
| `MaskSpanLength` | Gets or sets the mask span length. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `TopKLayers` | Gets or sets the number of top-k layers averaged for teacher targets. |
| `Variant` | Gets or sets the model variant ("base" or "large"). |

