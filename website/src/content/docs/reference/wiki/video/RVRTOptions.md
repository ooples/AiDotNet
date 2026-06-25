---
title: "RVRTOptions"
description: "Configuration options for the RVRT recurrent video restoration transformer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the RVRT recurrent video restoration transformer.

## For Beginners

RVRT processes video in small groups of frames at a time,
passing information forward like a memory. Within each group, it uses "guided
deformable attention" -- the model knows where to look in neighboring frames because
optical flow tells it where objects moved. This makes it both fast (small groups)
and accurate (guided by motion).

## How It Works

RVRT (Liang et al., NeurIPS 2022) combines recurrent processing with transformers:

- Recurrent frame grouping: processes video in overlapping clips of ClipSize frames,

with hidden states propagated between clips for long-range temporal modeling

- Guided deformable attention (GDA): attention offsets are guided by optical flow,

combining the efficiency of deformable attention with explicit motion information

- Multi-scale temporal fusion: features from different temporal scales are fused

through a hierarchical structure

- Applicable to multiple tasks: SR, deblurring, and denoising in one architecture

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RVRTOptions` | Initializes a new instance with default values. |
| `RVRTOptions(RVRTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClipSize` | Gets or sets the number of frames per recurrent clip. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBlocks` | Gets or sets the number of transformer blocks per stage. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFrameGroups` | Gets or sets the number of recurrent frame groups for temporal propagation. |
| `NumHeads` | Gets or sets the number of attention heads for guided deformable attention. |
| `NumSamplingPoints` | Gets or sets the number of sampling points per deformable attention head. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WindowSize` | Gets or sets the spatial window size for local attention. |

