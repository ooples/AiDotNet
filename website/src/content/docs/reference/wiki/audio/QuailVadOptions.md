---
title: "QuailVadOptions"
description: "Configuration options for the Quail VAD model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.VoiceActivity`

Configuration options for the Quail VAD model.

## For Beginners

Quail VAD detects when someone is speaking in audio - like a smart
"is anyone talking right now?" detector. It's designed to be small and fast enough to run
on phones and embedded devices in real-time.

## How It Works

Quail VAD (2024) is a lightweight voice activity detection model optimized for
on-device deployment. It uses a compact CNN-RNN architecture with knowledge distillation
from larger models, achieving high accuracy with minimal computational overhead.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FrameSizeMs` | Gets or sets the frame size in milliseconds. |
| `HiddenDim` | Gets or sets the hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MinSilenceDuration` | Gets or sets the minimum silence duration in seconds. |
| `MinSpeechDuration` | Gets or sets the minimum speech duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumCNNLayers` | Gets or sets the number of CNN layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `RNNHiddenSize` | Gets or sets the RNN hidden size. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the speech detection threshold. |
| `Variant` | Gets or sets the model variant. |

