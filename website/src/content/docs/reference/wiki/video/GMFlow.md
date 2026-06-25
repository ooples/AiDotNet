---
title: "GMFlow<T>"
description: "GMFlow (Global Matching Flow) for accurate optical flow estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Motion`

GMFlow (Global Matching Flow) for accurate optical flow estimation.

## For Beginners

GMFlow estimates how pixels move between video frames using
a global matching approach. Unlike local methods that only look at small neighborhoods,
GMFlow considers the entire image when matching pixels, making it better at:

- Large displacements (fast motion)
- Textureless regions
- Occlusions and disocclusions
- Repetitive patterns

The output is a "flow field" where each pixel has (dx, dy) values indicating
where that pixel moved to in the next frame.

## How It Works

**Technical Details:**

- Transformer-based global matching architecture
- Cross-attention for finding correspondences
- Hierarchical refinement for sub-pixel accuracy
- Self-attention for context aggregation

**Reference:** Xu et al., "GMFlow: Learning Optical Flow via Global Matching"
CVPR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GMFlow` | Initializes a new instance with default architecture settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputHeight` | Gets the input frame height. |
| `InputWidth` | Gets the input frame width. |
| `NumTransformerLayers` | Gets the number of transformer layers. |
| `SupportsTraining` | Gets whether training is supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAttention(Tensor<>,Tensor<>,Tensor<>)` | Applies scaled dot-product attention following the Transformer mechanism. |
| `EstimateBidirectionalFlow(Tensor<>,Tensor<>)` | Computes forward and backward flow for consistency checking. |
| `EstimateFlow(Tensor<>,Tensor<>)` | Estimates optical flow between two frames. |
| `EstimateFlowWithOcclusion(Tensor<>,Tensor<>)` | Estimates flow with occlusion mask. |
| `GetOptions` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `WarpImage(Tensor<>,Tensor<>)` | Warps an image using the estimated flow. |

