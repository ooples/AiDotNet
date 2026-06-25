---
title: "IFRNetOptions"
description: "Configuration options for the IFRNet intermediate feature refine network."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the IFRNet intermediate feature refine network.

## For Beginners

Most frame interpolation methods first estimate motion (optical flow),
then use it to warp frames. If the flow is wrong, the result is wrong. IFRNet takes a
different approach: it starts with a rough flow estimate, uses it to get initial features,
then directly refines those features to produce the final frame. This is more forgiving
of flow errors and produces sharper results.

## How It Works

IFRNet (Kong et al., CVPR 2022) uses coarse-to-fine intermediate feature refinement:

- Encoder-decoder with skip connections: shared encoder extracts multi-scale features

from both input frames, decoder progressively refines the interpolation result from
coarsest to finest scale

- Intermediate feature refinement (IFR): at each decoder level, instead of refining the

optical flow, IFRNet directly refines the intermediate features of the target frame,
avoiding error accumulation from flow estimation

- Coarse-to-fine architecture: 3-level pyramid where each level operates at half the

resolution of the next, with learned upsampling between levels

- Task-oriented flow: optical flow is used only as an initial guide for feature warping,

then discarded in favor of direct feature refinement

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IFRNetOptions` | Initializes a new instance with default values. |
| `IFRNetOptions(IFRNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels at the finest level. |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels. |
| `NumRefineBlocks` | Gets or sets the number of refinement blocks per decoder level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `UseTaskOrientedFlow` | Gets or sets whether to use task-oriented flow initialization. |
| `Variant` | Gets or sets the model variant. |

