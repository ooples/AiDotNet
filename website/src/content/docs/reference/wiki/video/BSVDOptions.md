---
title: "BSVDOptions"
description: "Configuration options for BSVD bidirectional streaming video denoising."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for BSVD bidirectional streaming video denoising.

## For Beginners

BSVD cleans up noisy video in real-time by looking at both past and
future frames. Unlike methods that need all frames at once, it processes them in a stream
using small memory buffers, making it practical for live video and long recordings.

## How It Works

BSVD (Qi et al., ACM MM 2022) enables real-time video denoising through bidirectional
streaming with efficient buffer management:

- Bidirectional streaming: processes video in both forward and backward passes with shared

buffers, so each frame benefits from both past and future context

- Streaming buffers: maintains compact latent buffers instead of storing full frames,

enabling constant-memory processing regardless of video length

- Real-time capability: designed for 30+ fps denoising on consumer GPUs through

efficient buffer reuse and single-pass-per-direction processing

- Noise-adaptive: handles varying noise levels without requiring noise-level input,

making it suitable for real-world video with spatially varying noise

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BSVDOptions` | Initializes a new instance with default values. |
| `BSVDOptions(BSVDOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferDim` | Gets or sets the hidden state dimension for streaming buffers. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumLevels` | Gets or sets the number of U-Net encoder/decoder levels. |
| `NumRecurrentBlocks` | Gets or sets the number of recurrent blocks per direction. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

