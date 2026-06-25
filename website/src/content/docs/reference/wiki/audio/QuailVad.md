---
title: "QuailVad<T>"
description: "Quail VAD - lightweight voice activity detection optimized for on-device deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.VoiceActivity`

Quail VAD - lightweight voice activity detection optimized for on-device deployment.

## For Beginners

Quail VAD is a lightweight "is someone talking?" detector that runs
efficiently on phones and small devices. Despite being much smaller than models like
Silero VAD, it achieves competitive accuracy by learning from larger, more powerful models
during training.

**Usage:**

## How It Works

Quail VAD (2024) is a compact CNN-RNN voice activity detector designed for edge devices.
It uses knowledge distillation from larger teacher models to maintain high accuracy while
keeping the model small enough for mobile phones and embedded systems. The model processes
short audio frames and outputs speech probabilities with minimal latency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuailVad(NeuralNetworkArchitecture<>,QuailVadOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Quail VAD model in native training mode. |
| `QuailVad(NeuralNetworkArchitecture<>,String,QuailVadOptions)` | Creates a Quail VAD model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameSize` |  |
| `MinSilenceDurationMs` |  |
| `MinSpeechDurationMs` |  |
| `SampleRate` |  |
| `Threshold` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectSpeech(Tensor<>)` |  |
| `DetectSpeechSegments(Tensor<>)` |  |
| `GetFrameProbabilities(Tensor<>)` |  |
| `GetSpeechProbability(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |

