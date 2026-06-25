---
title: "XVFIOptions"
description: "Configuration options for XVFI extreme video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for XVFI extreme video frame interpolation.

## For Beginners

XVFI is designed for extreme cases: very high resolution video (4K/8K)
where objects move very far between frames. It uses a multi-level approach that first captures
big movements, then progressively adds fine detail, enabling frame interpolation even when
objects move hundreds of pixels between frames.

## How It Works

XVFI (Sim et al., ICCV 2021) handles extreme motion for high-FPS video:

- Extreme motion handling: designed for 4K/8K video with very large frame-to-frame

displacements (100+ pixels), far beyond what standard flow networks can handle

- Complementary flow: estimates both global (affine) and local (dense) optical flow fields,

combining them with learned blending weights so global flow handles camera motion and
local flow handles object motion

- Multi-scale architecture: a 7-level feature pyramid with flow estimation at each scale,

starting from 1/64 resolution for very large motions and refining up to full resolution

- Bilinear flow upsampling: uses learned bilinear upsampling kernels (not fixed bilinear

interpolation) to upsample flow fields between pyramid levels, preserving sharp motion
boundaries during upsampling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XVFIOptions` | Initializes a new instance with default values. |
| `XVFIOptions(XVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAffineParams` | Gets or sets the number of global flow affine parameters. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels. |
| `NumResBlocks` | Gets or sets the number of residual blocks per level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `UseComplementaryFlow` | Gets or sets whether to use complementary flow (global + local). |
| `Variant` | Gets or sets the model variant. |

