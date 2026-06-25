---
title: "PengiOptions"
description: "Configuration options for the Pengi model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Multimodal`

Configuration options for the Pengi model.

## For Beginners

Pengi treats all audio understanding as a conversation. Instead of
having separate models for "what sound is this?" and "describe this audio", Pengi uses one
model that can answer any question about audio by generating text responses.

## How It Works

Pengi (Deshmukh et al., 2023, Microsoft) is an audio language model that frames all audio
tasks as text-generation tasks. It uses a frozen audio encoder (Audio Spectrogram Transformer)
paired with a pre-trained language model, enabling open-ended audio reasoning, captioning,
and question answering without task-specific heads.

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioEncoderDim` | Gets or sets the audio encoder dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LLMHiddenDim` | Gets or sets the LLM hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxAudioDurationSeconds` | Gets or sets the maximum audio duration in seconds. |
| `MaxResponseTokens` | Gets or sets the maximum response tokens. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumProjectionLayers` | Gets or sets the number of projection layers from audio to LLM space. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant. |

