---
title: "MarbleNet<T>"
description: "MarbleNet lightweight 1D separable convolutional VAD model (NVIDIA NeMo)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.VoiceActivity`

MarbleNet lightweight 1D separable convolutional VAD model (NVIDIA NeMo).

## For Beginners

MarbleNet is NVIDIA's efficient voice activity detector. It uses a
special neural network layer (separable convolutions) that makes it very fast while still
being accurate. Think of it as a "speech or not?" classifier that can run in real-time
even on a phone or small device.

**Usage:**

## How It Works

MarbleNet (Jia et al., 2021, NVIDIA NeMo) is a lightweight 1D time-channel separable
convolutional model for voice activity detection. It uses depth-wise separable convolutions
with sub-word modeling to achieve state-of-the-art accuracy while being fast enough for
real-time streaming on edge devices.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MarbleNet(NeuralNetworkArchitecture<>,MarbleNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a MarbleNet model in native training mode. |
| `MarbleNet(NeuralNetworkArchitecture<>,String,MarbleNetOptions)` | Creates a MarbleNet model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FrameSize` |  |
| `MinSilenceDurationMs` |  |
| `MinSpeechDurationMs` |  |
| `Threshold` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IVoiceActivityDetector{T}#ResetState` |  |
| `DetectSpeech(Tensor<>)` |  |
| `DetectSpeechSegments(Tensor<>)` |  |
| `GetFrameProbabilities(Tensor<>)` |  |
| `GetSpeechProbability(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |

