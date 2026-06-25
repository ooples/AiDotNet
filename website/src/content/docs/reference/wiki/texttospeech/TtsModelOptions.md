---
title: "TtsModelOptions"
description: "Base configuration options for text-to-speech models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech`

Base configuration options for text-to-speech models.

## For Beginners

These options configure the TtsModel model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TtsModelOptions` | Initializes a new instance with default values. |
| `TtsModelOptions(TtsModelOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FftSize` | Gets or sets the FFT window size. |
| `HiddenDim` | Gets or sets the model hidden dimension. |
| `HopSize` | Gets or sets the hop size for mel-spectrogram computation. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxMelLength` | Gets or sets the maximum output audio length in mel frames. |
| `MaxTextLength` | Gets or sets the maximum input text length. |
| `MelChannels` | Gets or sets the number of mel-spectrogram frequency channels. |
| `ModelPath` | Gets or sets the ONNX model path. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `VocabSize` | Gets or sets the phoneme/text vocabulary size. |
| `WeightDecay` | Gets or sets the weight decay. |

