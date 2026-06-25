---
title: "Wav2Vec2Options"
description: "Configuration options for the wav2vec 2.0 self-supervised speech model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Foundations`

Configuration options for the wav2vec 2.0 self-supervised speech model.

## For Beginners

wav2vec 2.0 pioneered self-supervised learning for speech. It
listens to raw audio, masks parts of it, and learns to predict the missing parts. This
teaches it a deep understanding of speech that can then be used for tasks like
transcription with very little labeled data.

## How It Works

wav2vec 2.0 (Baevski et al., 2020, Meta) learns speech representations via contrastive
learning over quantized speech units. Pre-trained on 960 hours of LibriSpeech, it achieves
WER 1.8% on test-clean with only 10 minutes of labeled data when fine-tuned for ASR.

## Properties

| Property | Summary |
|:-----|:--------|
| `ContrastiveTemperature` | Gets or sets the contrastive loss temperature. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FeatureEncoderDim` | Gets or sets the CNN feature encoder output dimension. |
| `FeedForwardDim` | Gets or sets the feed-forward dimension. |
| `HiddenDim` | Gets or sets the transformer hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumQuantizationGroups` | Gets or sets the number of quantization codebooks for contrastive learning. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `QuantizationCodebookSize` | Gets or sets the quantization codebook size. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant ("base" or "large"). |

