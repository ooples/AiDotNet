---
title: "IFRNet<T>"
description: "IFRNet: intermediate feature refine network for efficient frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

IFRNet: intermediate feature refine network for efficient frame interpolation.

## For Beginners

Most frame interpolation methods first estimate motion (optical flow),
then use it to warp frames. If the flow is wrong, the result is wrong. IFRNet takes a
different approach: it starts with a rough flow estimate, uses it to get initial features,
then directly refines those features to produce the final frame. This is more forgiving
of flow errors and produces sharper results with fewer artifacts.

**Usage:**

## How It Works

IFRNet (Kong et al., CVPR 2022) uses coarse-to-fine intermediate feature refinement:

- Encoder-decoder with skip connections: shared encoder extracts multi-scale features

from both input frames, decoder progressively refines the interpolation result from
coarsest to finest scale

- Intermediate feature refinement (IFR): at each decoder level, instead of refining the

optical flow, IFRNet directly refines the intermediate features of the target frame,
avoiding error accumulation from iterated flow estimation

- Coarse-to-fine pyramid: 3-level pyramid where each level operates at half the resolution

of the next, with learned upsampling and skip connections between levels

- Task-oriented flow: optical flow is used only as an initial guide for feature warping,

then discarded in favor of direct feature refinement which is more robust to flow errors

**Reference:** "IFRNet: Intermediate Feature Refine Network for Efficient Frame
Interpolation" (Kong et al., CVPR 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IFRNet(NeuralNetworkArchitecture<>,IFRNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an IFRNet model in native training mode. |
| `IFRNet(NeuralNetworkArchitecture<>,String,IFRNetOptions)` | Creates an IFRNet model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

