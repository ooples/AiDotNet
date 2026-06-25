---
title: "TTVSROptions"
description: "Configuration options for the TTVSR trajectory-aware transformer for video SR."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the TTVSR trajectory-aware transformer for video SR.

## For Beginners

Imagine following a ball as it moves across frames. Instead
of looking at the same spot in every frame, TTVSR tracks where objects actually
go and gathers information along their path. This is much more effective than
fixed-position alignment because real video has complex motion.

## How It Works

TTVSR (Liu et al., ECCV 2022) tracks feature trajectories across frames:

- Trajectory-aware attention: instead of attending to fixed spatial locations,

attention follows estimated motion trajectories so features are gathered along
the path an object actually moved

- Cross-scale feature tokenization: visual tokens are extracted at multiple scales

and fused, capturing both fine texture and coarse structure

- Location map: a learned spatial map that helps the transformer locate the

trajectory routing positions across frames

- Long-range temporal modeling: trajectories span the full video length, not just

adjacent frames, enabling information flow from distant frames

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TTVSROptions` | Initializes a new instance with default values. |
| `TTVSROptions(TTVSROptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the reconstruction module. |
| `NumScales` | Gets or sets the number of cross-scale feature tokenization levels. |
| `NumTransformerBlocks` | Gets or sets the number of trajectory-aware transformer blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `ScaleFactor` | Gets or sets the spatial upscaling factor. |
| `TrajectoryLength` | Gets or sets the trajectory length (number of frames tracked per trajectory). |
| `Variant` | Gets or sets the model variant. |

