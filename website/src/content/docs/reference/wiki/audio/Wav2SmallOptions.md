---
title: "Wav2SmallOptions"
description: "Configuration options for the Wav2Small speech emotion recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Emotion`

Configuration options for the Wav2Small speech emotion recognition model.

## For Beginners

Wav2Small is like a student who learned from a large, expert teacher.
The teacher (wav2vec 2.0) is too big to run on phones or embedded devices, so Wav2Small
learns the teacher's emotion detection skills in a much smaller model. The result is fast,
accurate emotion detection that can run on resource-limited devices.

## How It Works

Wav2Small (Gomez-Alanis et al., 2024) is a lightweight speech emotion recognition model
that distills knowledge from large wav2vec 2.0 models into a compact architecture. It achieves
competitive accuracy on IEMOCAP and RAVDESS while being small enough for edge deployment.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationTemperature` | Gets or sets the knowledge distillation temperature. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmotionLabels` | Gets or sets the emotion label names. |
| `FeatureEncoderDim` | Gets or sets the CNN feature encoder output dimension. |
| `FeedForwardDim` | Gets or sets the feed-forward hidden dimension. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the hidden dimension of the compact encoder. |
| `HopLength` | Gets or sets the hop length between frames. |
| `IncludeArousalValence` | Gets or sets whether to include arousal/valence estimation. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumClasses` | Gets or sets the number of emotion classes. |
| `NumLayers` | Gets or sets the number of encoder layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |

