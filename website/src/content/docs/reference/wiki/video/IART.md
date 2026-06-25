---
title: "IART<T>"
description: "IART: implicit resampling-based alignment transformer for video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

IART: implicit resampling-based alignment transformer for video super-resolution.

## For Beginners

When aligning video frames, most models "warp" one frame to match
another using a grid. This can blur fine details because pixel positions don't perfectly
line up with grid points. IART solves this by learning a continuous function that can
read features at ANY position (not just grid points), like being able to zoom into
any spot on a map with perfect clarity rather than being limited to the printed grid.

**Usage:**

## How It Works

IART (Kai et al., CVPR 2024 Highlight) uses implicit neural representations for alignment:

- Implicit resampling: instead of warping features to discrete grid positions (which

causes interpolation artifacts), IART uses a continuous implicit function to sample
features at arbitrary sub-pixel positions with learned kernels

- Alignment transformer: cross-attention between reference and supporting frames where

sampling positions are offset by flow-guided implicit coordinates, achieving sub-pixel
accurate alignment without grid discretization

- Multi-scale implicit alignment: alignment at multiple feature resolutions, from

coarse structural alignment to fine texture-level resampling

- High-frequency preservation: the implicit function preserves sharp edges, thin lines,

and fine textures that grid-based bilinear/bicubic warping typically blurs

**Reference:** "IART: Implicit Resampling-based Alignment Transformer for Video
Super-Resolution" (Kai et al., CVPR 2024 Highlight)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IART(NeuralNetworkArchitecture<>,IARTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an IART model in native training mode. |
| `IART(NeuralNetworkArchitecture<>,String,IARTOptions)` | Creates an IART model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

