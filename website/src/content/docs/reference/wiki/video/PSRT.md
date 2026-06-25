---
title: "PSRT<T>"
description: "PSRT: progressive spatio-temporal alignment with window-based attention for video SR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

PSRT: progressive spatio-temporal alignment with window-based attention for video SR.

## For Beginners

PSRT aligns video frames step by step, starting with big motion
corrections (like camera shake) and progressively refining small details (sub-pixel
alignment). It uses "windows" in both space and time to efficiently find matching regions
across frames, similar to how Swin Transformer works but extended to 3D video volumes.

**Usage:**

## How It Works

PSRT (Shi et al., 2022) uses progressive window-based spatio-temporal attention:

- Spatio-temporal attention blocks (STABs): joint spatial and temporal attention within

3D windows (height x width x time) for aligned multi-frame feature fusion

- Progressive alignment: a coarse-to-fine encoder-decoder structure where early layers

capture large motions with downsampled features, and later layers refine sub-pixel
alignment at full resolution

- Window-based attention: limits attention to local spatio-temporal windows with shifted

window partitioning for cross-window information flow (Swin-style)

- Temporal mutual attention: cross-attention between the reference frame and each

supporting frame to explicitly align temporal features

**Reference:** "PSRT: Progressive Spatio-temporal Alignment for Video
Super-Resolution" (Shi et al., 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PSRT(NeuralNetworkArchitecture<>,PSRTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a PSRT model in native training mode. |
| `PSRT(NeuralNetworkArchitecture<>,String,PSRTOptions)` | Creates a PSRT model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

