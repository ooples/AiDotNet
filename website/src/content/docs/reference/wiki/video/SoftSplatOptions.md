---
title: "SoftSplatOptions"
description: "Configuration options for SoftSplat softmax splatting for video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for SoftSplat softmax splatting for video frame interpolation.

## For Beginners

When you push pixels from a source frame to a new position (forward
warping), multiple pixels might land in the same spot. SoftSplat uses a smart voting system
(softmax) where each pixel gets a learned "importance score" to decide which pixel wins
when there's a conflict, naturally handling which objects appear in front of others.

## How It Works

SoftSplat (Niklaus & Liu, CVPR 2020) uses softmax splatting for forward warping:

- Forward warping with softmax: instead of backward warping (which creates holes at

disocclusions), SoftSplat uses forward warping where source pixels are "splatted" to
target positions, with conflicts resolved via softmax weighting

- Importance metric Z: each source pixel carries a learned importance metric Z that controls

its softmax weight during splatting, allowing the model to automatically learn that
foreground (closer) objects should occlude background (farther) objects

- Feature-space splatting: splatting is performed not on raw pixels but on deep feature maps,

allowing the synthesis network to work with rich feature representations and produce
higher-quality output

- Synthesis network: a GridNet-style synthesis network takes the splatted feature maps and

produces the final interpolated frame, handling residual refinement and artifact removal

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoftSplatOptions` | Initializes a new instance with default values. |
| `SoftSplatOptions(SoftSplatOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatureBlocks` | Gets or sets the feature extraction depth (number of VGG-like blocks). |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumGridNetLevels` | Gets or sets the number of GridNet synthesis levels. |
| `NumResBlocksPerRow` | Gets or sets the number of residual blocks per GridNet row. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `UseImportanceMetric` | Gets or sets whether to use the learned importance metric Z. |
| `Variant` | Gets or sets the model variant. |

