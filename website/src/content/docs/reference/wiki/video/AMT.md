---
title: "AMT<T>"
description: "AMT: all-pairs multi-field transforms for efficient frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

AMT: all-pairs multi-field transforms for efficient frame interpolation.

## For Beginners

AMT tries every possible match between pixels in two frames
(all-pairs). For each pixel, instead of guessing a single motion direction, it proposes
several candidates and lets the network pick the best one. This handles tricky cases
like objects moving in front of each other (occlusion) or objects appearing from behind.

**Usage:**

## How It Works

AMT (Li et al., CVPR 2023) uses correlation-based all-pairs multi-field transforms:

- All-pairs correlation: computes dense 4D cost volume between every pixel pair across

two frames at multiple scales, providing exhaustive motion correspondence information
that captures all possible matches including occluded and newly-visible regions

- Multi-field transforms: instead of a single flow field, predicts K candidate flow

fields per pixel, each capturing a plausible motion hypothesis, which are merged via
learned soft selection weights for robust motion estimation

- Iterative GRU refinement: coarse-to-fine correlation lookup with GRU-based iterative

updates that progressively refine the multi-field estimates over N iterations

- Efficient separable correlation: uses separable 1D correlation (H then W) instead of

full 2D correlation, reducing the quartic O(N^4) cost to quadratic O(N^2)

**Reference:** "AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation"
(Li et al., CVPR 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AMT(NeuralNetworkArchitecture<>,AMTOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an AMT model in native training mode. |
| `AMT(NeuralNetworkArchitecture<>,String,AMTOptions)` | Creates an AMT model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

