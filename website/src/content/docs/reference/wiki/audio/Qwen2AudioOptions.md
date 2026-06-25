---
title: "Qwen2AudioOptions"
description: "Configuration options for the Qwen2-Audio multimodal audio-language model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Multimodal`

Configuration options for the Qwen2-Audio multimodal audio-language model.

## For Beginners

Qwen2-Audio is like having a conversation partner who can hear.
You play it audio and ask questions like "What sounds do you hear?" or "Describe this music."
It uses a powerful language model (similar to ChatGPT) combined with an audio understanding
system to provide intelligent responses about any audio input.

## How It Works

Qwen2-Audio (Chu et al., 2024, Alibaba) is a large audio-language model that can process
audio and text inputs to generate text responses. It uses a Whisper-style audio encoder
with a Qwen2 language model backbone, connected by a perceiver-style adapter. It supports
audio captioning, question answering, sound event detection, and audio reasoning.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterDim` | Gets or sets the perceiver adapter output dimension. |
| `AudioEncoderDim` | Gets or sets the audio encoder dimension (Whisper-style). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LMHiddenDim` | Gets or sets the language model hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxAudioDurationSeconds` | Gets or sets the maximum audio duration in seconds. |
| `MaxResponseTokens` | Gets or sets the maximum response length in tokens. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAudioEncoderHeads` | Gets or sets the number of audio encoder attention heads. |
| `NumAudioEncoderLayers` | Gets or sets the number of audio encoder layers. |
| `NumLMHeads` | Gets or sets the number of language model attention heads. |
| `NumLMLayers` | Gets or sets the number of language model layers. |
| `NumLatentTokens` | Gets or sets the number of perceiver latent tokens. |
| `NumMels` | Gets or sets the number of mel spectrogram channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Temperature` | Gets or sets the sampling temperature. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |
| `VocabSize` | Gets or sets the LM vocabulary size. |

