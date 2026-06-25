---
title: "MarbleNetOptions"
description: "Configuration options for the MarbleNet voice activity detection model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.VoiceActivity`

Configuration options for the MarbleNet voice activity detection model.

## For Beginners

MarbleNet is NVIDIA's efficient voice activity detector. It uses a
special type of neural network layer (separable convolutions) that makes it very fast while
still being accurate. Think of it as a "speech or not?" classifier that can run in real-time
even on a phone or small device.

## How It Works

MarbleNet (Jia et al., 2021, NVIDIA NeMo) is a lightweight 1D time-channel separable
convolutional model for voice activity detection. It uses depth-wise separable convolutions
with sub-word modeling to achieve state-of-the-art accuracy while being fast enough for
real-time streaming on edge devices.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `FftSize` | Gets or sets the FFT window size. |
| `FrameDurationMs` | Gets or sets the frame duration in milliseconds. |
| `HopLength` | Gets or sets the hop length between frames. |
| `InitialFilters` | Gets or sets the number of initial convolution filters. |
| `KernelSize` | Gets or sets the kernel size for separable convolutions. |
| `LearningRate` | Gets or sets the learning rate. |
| `MinSilenceDurationMs` | Gets or sets the minimum silence duration in milliseconds. |
| `MinSpeechDurationMs` | Gets or sets the minimum speech duration in milliseconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBlocks` | Gets or sets the number of separable conv blocks. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `SubBlocksPerBlock` | Gets or sets the number of sub-blocks per block. |
| `Threshold` | Gets or sets the detection threshold. |

