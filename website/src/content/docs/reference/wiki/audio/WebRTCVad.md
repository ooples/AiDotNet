---
title: "WebRTCVad<T>"
description: "Neural WebRTC VAD model for low-latency voice activity detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.VoiceActivity`

Neural WebRTC VAD model for low-latency voice activity detection.

## For Beginners

WebRTC VAD is a very fast "is someone talking?" detector used in
video calls and voice chat. It processes tiny chunks of audio (10-30 milliseconds) and
instantly decides if speech is present.

**Usage:**

## How It Works

WebRTC VAD is a lightweight voice activity detection model inspired by the GMM-based
detector in the WebRTC framework but reimplemented as a neural network for improved accuracy.
It operates at very low latency (10-30ms frames) and is designed for real-time communication.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WebRTCVad(NeuralNetworkArchitecture<>,String,WebRTCVadOptions)` | Creates a WebRTC VAD model in ONNX inference mode. |
| `WebRTCVad(NeuralNetworkArchitecture<>,WebRTCVadOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a WebRTC VAD model in native training mode. |

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

