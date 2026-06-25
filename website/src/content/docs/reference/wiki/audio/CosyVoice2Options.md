---
title: "CosyVoice2Options"
description: "Configuration options for the CosyVoice2 model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.TextToSpeech`

Configuration options for the CosyVoice2 model.

## For Beginners

CosyVoice2 converts text to natural-sounding speech that can
clone any voice from just a few seconds of reference audio. It's fast enough for
real-time applications like voice assistants and audiobooks.

## How It Works

CosyVoice2 (Du et al., 2024, Alibaba) is a scalable streaming TTS model that achieves
natural-sounding speech with very low latency. It uses a finite scalar quantization (FSQ)
codec with a flow-matching decoder and supports zero-shot voice cloning, cross-lingual
synthesis, and emotion control.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecoderDim` | Gets or sets the decoder dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumMels` | Gets or sets the number of mel bins. |
| `NumTextEncoderLayers` | Gets or sets the number of text encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the output sample rate in Hz. |
| `SpeakerEmbeddingDim` | Gets or sets the speaker embedding dimension. |
| `TextEncoderDim` | Gets or sets the text encoder dimension. |
| `Variant` | Gets or sets the model variant. |

