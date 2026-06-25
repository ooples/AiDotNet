---
title: "DRVIOptions"
description: "Configuration options for the DRVI disentangled representations model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the DRVI disentangled representations model.

## For Beginners

DRVI separates "what things look like" from "how things move".
By processing appearance and motion independently, it can better handle cases where
objects look similar but move differently, or where the same object appears in different
lighting conditions across frames.

## How It Works

DRVI (2024) disentangles content and motion for video interpolation:

- Disentangled encoders: separate encoders for content (appearance, texture, color) and

motion (displacement, deformation) that process frame pairs independently

- Content encoder: extracts appearance features invariant to motion, shared across all

timesteps so the model doesn't re-extract appearance at each interpolation point

- Motion encoder: captures inter-frame displacement fields at multiple scales, enabling

the model to handle both global camera motion and local object motion

- Disentangled decoder: recombines content and motion representations with learned gating

at each scale, allowing fine control over which content features are warped by which
motion components

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DRVIOptions` | Initializes a new instance with default values. |
| `DRVIOptions(DRVIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumContentBlocks` | Gets or sets the number of content encoder blocks. |
| `NumDecoderBlocks` | Gets or sets the number of decoder blocks for recombination. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumMotionBlocks` | Gets or sets the number of motion encoder blocks. |
| `NumScales` | Gets or sets the number of pyramid scales. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

