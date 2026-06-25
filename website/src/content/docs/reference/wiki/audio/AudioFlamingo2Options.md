---
title: "AudioFlamingo2Options"
description: "Configuration options for the Audio Flamingo 2 model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Multimodal`

Configuration options for the Audio Flamingo 2 model.

## For Beginners

Audio Flamingo 2 is like giving a language AI the ability to hear.
It can listen to audio recordings and answer questions about them, generate descriptions,
or reason about what's happening in the audio scene.

## How It Works

Audio Flamingo 2 (2024) extends the Flamingo architecture for audio understanding with
interleaved audio-text inputs. It uses a frozen audio encoder with perceiver-style
cross-attention to adapt a pre-trained LLM for audio captioning, QA, and reasoning.

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
| `NumPerceiverLayers` | Gets or sets the number of perceiver layers. |
| `NumPerceiverTokens` | Gets or sets the number of perceiver latent tokens. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Variant` | Gets or sets the model variant. |

