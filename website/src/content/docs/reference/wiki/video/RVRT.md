---
title: "RVRT<T>"
description: "RVRT: recurrent video restoration transformer with guided deformable attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

RVRT: recurrent video restoration transformer with guided deformable attention.

## For Beginners

RVRT processes video in small groups of frames at a time,
passing "memory" forward so later frames benefit from earlier ones. Within each group,
it uses "guided deformable attention" -- the model knows roughly where to look in
neighboring frames (from optical flow), then fine-tunes those positions with learned
offsets. This combines the speed of small groups with the accuracy of motion guidance.

**Usage:**

## How It Works

RVRT (Liang et al., NeurIPS 2022) combines recurrent processing with transformers:

- Recurrent frame grouping: processes video in overlapping clips of ClipSize frames,

propagating hidden state features between clips for long-range temporal context

- Guided deformable attention (GDA): attention sampling offsets are initialized from

optical flow estimates, combining the flexibility of deformable attention with
explicit motion guidance for more accurate alignment

- Multi-scale temporal fusion: features from different temporal scales (clip-level

and global recurrence) are hierarchically merged

- Versatile restoration: the same architecture handles SR, deblurring, and denoising

by changing only the loss function and training data

**Reference:** "RVRT: Recurrent Video Restoration Transformer" (Liang et al., NeurIPS 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RVRT(NeuralNetworkArchitecture<>,RVRTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a RVRT model in native training mode. |
| `RVRT(NeuralNetworkArchitecture<>,String,RVRTOptions)` | Creates a RVRT model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

