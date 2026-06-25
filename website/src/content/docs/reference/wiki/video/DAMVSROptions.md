---
title: "DAMVSROptions"
description: "Configuration options for the DAM-VSR appearance-motion disentangled video super-resolution model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the DAM-VSR appearance-motion disentangled video super-resolution model.

## For Beginners

Most video SR models mix up "what things look like" (appearance) with
"how things move" (motion). DAM-VSR separates these into two branches so each can focus
on what it does best, then combines them for the final high-resolution output.

## How It Works

DAM-VSR (SIGGRAPH 2025) disentangles appearance and motion for cleaner video SR:

- Appearance branch: extracts texture and structure features from individual frames
- Motion branch: captures temporal dynamics and inter-frame correspondences separately
- Appearance-Motion Fusion: combines both branches with learned gating for reconstruction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DAMVSROptions` | Initializes a new instance with default values. |
| `DAMVSROptions(DAMVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DeformableGroups` | Gets or sets the number of deformable groups per attention head. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFrames` | Gets or sets the number of input frames. |
| `NumHeads` | Gets or sets the number of attention heads in the motion branch. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the reconstruction module. |
| `NumSamplingPoints` | Gets or sets the number of sampling points per deformable attention head. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |
| `WarmupSteps` | Gets or sets the warmup steps. |

