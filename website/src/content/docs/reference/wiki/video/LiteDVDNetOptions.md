---
title: "LiteDVDNetOptions"
description: "Configuration options for LiteDVDNet lightweight deep video denoising."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for LiteDVDNet lightweight deep video denoising.

## For Beginners

LiteDVDNet is a fast, lightweight video denoiser. It first cleans
each frame individually, then combines information from nearby frames to improve quality.
It's designed to run efficiently on devices with limited computing power.

## How It Works

LiteDVDNet is a lightweight variant of DVDNet for efficient video denoising:

- Two-stage pipeline: first stage denoises each frame independently, second stage

fuses temporal information from the independently denoised frames

- Lightweight blocks: uses depthwise separable convolutions instead of standard

convolutions, reducing parameters by 8-10x while maintaining quality

- Non-blind support: accepts noise level sigma as input, allowing the network to

adapt its denoising strength to the actual noise level

- Efficient fusion: simple temporal fusion via 1x1 convolutions over stacked frames

rather than expensive optical flow or attention mechanisms

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LiteDVDNetOptions` | Initializes a new instance with default values. |
| `LiteDVDNetOptions(LiteDVDNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `ExpansionFactor` | Gets or sets the depthwise separable expansion factor. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBlocks` | Gets or sets the number of denoising blocks per stage. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `TemporalWindowSize` | Gets or sets the temporal window size (number of input frames). |
| `Variant` | Gets or sets the model variant. |

