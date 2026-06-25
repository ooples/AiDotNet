---
title: "HuBERTOptions"
description: "Configuration options for the HuBERT (Hidden-Unit BERT) self-supervised speech model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Foundations`

Configuration options for the HuBERT (Hidden-Unit BERT) self-supervised speech model.

## For Beginners

HuBERT learns to understand speech by listening to millions of
hours of audio. It predicts hidden "units" in speech (like phonemes) without any labels.
After pre-training, it can be fine-tuned for tasks like transcription or speaker ID.

## How It Works

HuBERT (Hsu et al., 2021, Meta) learns speech representations by predicting masked
discrete speech units derived from clustering. It achieves strong performance on speech
recognition, speaker verification, and emotion detection when fine-tuned.

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
| `Variant` | Gets or sets the model variant ("base" or "large"). |

