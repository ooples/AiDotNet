---
title: "VFITOptions"
description: "Configuration options for VFIT video frame interpolation transformer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for VFIT video frame interpolation transformer.

## For Beginners

VFIT uses more than just two frames to create the in-between frame.
By looking at a wider window of frames (2 before and 2 after), it can better understand
complex motions like acceleration, deceleration, and periodic movements.

## How It Works

VFIT (Shi et al., CVPR 2022) uses vision transformers for multi-frame interpolation:

- Multi-frame input: takes multiple input frames (typically 4: two before and two after the

target) to provide richer temporal context than 2-frame methods

- Temporal transformer: applies temporal self-attention across the multiple input frames,

learning long-range temporal dependencies and motion patterns that span multiple frames

- Spatial-temporal factorization: factorizes the full 3D attention into separate spatial

(within each frame) and temporal (across frames) attention for efficiency

- Progressive synthesis: generates the intermediate frame progressively from coarse to fine,

with transformer attention applied at each resolution level

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VFITOptions` | Initializes a new instance with default values. |
| `VFITOptions(VFITOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumInputFrames` | Gets or sets the number of input frames (typically 4). |
| `NumSpatialLayers` | Gets or sets the number of spatial transformer layers. |
| `NumTemporalLayers` | Gets or sets the number of temporal transformer layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

