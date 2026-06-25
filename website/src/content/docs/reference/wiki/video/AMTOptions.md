---
title: "AMTOptions"
description: "Configuration options for the AMT all-pairs multi-field transforms model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the AMT all-pairs multi-field transforms model.

## For Beginners

AMT tries every possible match between pixels in two frames
(all-pairs). For each pixel, instead of guessing a single motion direction, it proposes
several candidates and lets the network pick the best one. This handles tricky cases
like objects moving in front of each other or disappearing behind things.

## How It Works

AMT (Li et al., CVPR 2023) uses correlation-based all-pairs transforms:

- All-pairs correlation: computes dense 4D cost volume between every pixel pair across

two frames at multiple scales, providing exhaustive motion correspondence information

- Multi-field transforms: instead of a single flow field, predicts multiple (K) candidate

flow fields per pixel, each capturing a plausible motion hypothesis, which are then
merged via learned soft selection weights

- Iterative refinement: coarse-to-fine correlation lookup with GRU-based iterative

updates that progressively refine the multi-field estimates

- Efficient correlation: uses separable 1D correlation (H then W) instead of full 2D

correlation to reduce the quartic cost to quadratic

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AMTOptions` | Initializes a new instance with default values. |
| `AMTOptions(AMTOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CorrelationRadius` | Gets or sets the correlation search radius at each level. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumCorrelationLevels` | Gets or sets the number of correlation pyramid levels. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFlowFields` | Gets or sets the number of candidate flow fields per pixel. |
| `NumRefinementIters` | Gets or sets the number of GRU refinement iterations. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

