---
title: "WebRTCVadOptions"
description: "Configuration options for the WebRTC VAD neural model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.VoiceActivity`

Configuration options for the WebRTC VAD neural model.

## For Beginners

WebRTC VAD is a very fast "is someone talking?" detector used in
video calls and voice chat. It processes tiny chunks of audio (10-30 milliseconds) and
instantly decides if speech is present. Speed matters most here - it needs to work in
real-time without adding noticeable delay to your call.

## How It Works

WebRTC VAD is a lightweight voice activity detection model inspired by the GMM-based
detector in the WebRTC framework but reimplemented as a neural network for improved accuracy.
It operates at very low latency (10-30ms frames) and is designed for real-time communication.

## Properties

| Property | Summary |
|:-----|:--------|
| `AggressivenessMode` | Gets or sets the aggressiveness mode (0-3, higher = more aggressive filtering). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FrameDurationMs` | Gets or sets the frame duration in milliseconds (10, 20, or 30). |
| `HiddenDim` | Gets or sets the hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MinSilenceDurationMs` | Gets or sets the minimum silence duration in milliseconds. |
| `MinSpeechDurationMs` | Gets or sets the minimum speech duration in milliseconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumLayers` | Gets or sets the number of encoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `Threshold` | Gets or sets the detection threshold. |

