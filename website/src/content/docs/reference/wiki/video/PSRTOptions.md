---
title: "PSRTOptions"
description: "Configuration options for the PSRT progressive spatio-temporal alignment model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the PSRT progressive spatio-temporal alignment model.

## For Beginners

PSRT aligns video frames step by step, starting with big motion
corrections and progressively refining small details. It uses "windows" (small patches)
in both space and time to efficiently find corresponding regions across frames, similar
to how Swin Transformer works but extended to video.

## How It Works

PSRT (Shi et al., 2022) uses progressive window-based spatio-temporal attention:

- Spatio-temporal attention blocks (STABs): joint spatial and temporal attention within

3D windows (height x width x time) for aligned multi-frame feature fusion

- Progressive alignment: a coarse-to-fine encoder-decoder structure where early layers

capture large motions and later layers refine sub-pixel alignment

- Window-based attention: limits attention to local spatio-temporal windows for

computational efficiency while shifted windows enable cross-window communication

- Temporal mutual attention: cross-attention between reference and supporting frames

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PSRTOptions` | Initializes a new instance with default values. |
| `PSRTOptions(PSRTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumSTABs` | Gets or sets the number of spatio-temporal attention blocks (STABs). |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `TemporalRadius` | Gets or sets the temporal window radius (number of neighboring frames). |
| `Variant` | Gets or sets the model variant. |
| `WindowSize` | Gets or sets the spatial window size for window-based attention. |

