---
title: "IARTOptions"
description: "Configuration options for the IART implicit resampling-based alignment transformer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the IART implicit resampling-based alignment transformer.

## For Beginners

When aligning video frames, most models "warp" one frame to match
another using a grid. This can blur fine details because pixel positions don't perfectly
line up. IART solves this by using a continuous function that can sample features at
any position (not just grid points), preserving sharp edges and textures that other
methods would smooth out.

## How It Works

IART (Kai et al., CVPR 2024 Highlight) uses implicit neural representations for alignment:

- Implicit resampling: instead of warping features to discrete grid positions (which

causes interpolation artifacts), IART uses a continuous implicit function to resample
features at arbitrary sub-pixel positions with learned kernels

- Alignment transformer: cross-attention between the reference frame and supporting

frames, where the attention sampling positions are offset by flow-guided implicit
coordinates rather than fixed grid positions

- Multi-scale implicit alignment: alignment at multiple feature resolutions, from

coarse structural alignment to fine texture-level resampling

- Preserves high-frequency details that grid-based warping typically blurs

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IARTOptions` | Initializes a new instance with default values. |
| `IARTOptions(IARTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `ImplicitDim` | Gets or sets the dimension of the implicit coordinate embedding. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumResBlocks` | Gets or sets the number of residual blocks in reconstruction. |
| `NumScales` | Gets or sets the number of implicit resampling scales. |
| `NumTransformerBlocks` | Gets or sets the number of alignment transformer blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `Variant` | Gets or sets the model variant. |

