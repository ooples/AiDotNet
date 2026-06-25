---
title: "SegmentationTensorOps"
description: "Static helper methods for common segmentation tensor operations (argmax, softmax, etc.) used by models implementing segmentation interfaces."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.ComputerVision.Segmentation.Common`

Static helper methods for common segmentation tensor operations (argmax, softmax, etc.)
used by models implementing segmentation interfaces.

## Methods

| Method | Summary |
|:-----|:--------|
| `ArgmaxAlongClassDim(Tensor<>)` | Computes argmax along the class dimension, producing a per-pixel class index map. |
| `BoxMask(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates a spatial mask that is 1.0 inside the box [x1, y1, x2, y2) and 0.0 outside. |
| `EnsureUnbatched(Tensor<>)` | Removes the batch dimension from a tensor if present, keeping only batch index 0. |
| `GaussianMask(Int32,Int32,Double,Double,Double)` | Generates a 2D Gaussian attention mask centered at (cx, cy) with given sigma. |
| `LabelConnectedComponents(Tensor<>,Int32)` | Labels connected components of pixels matching a target class in the class map. |
| `PixelAffinity(Tensor<>,Tensor<>)` | Computes per-pixel cosine similarity between two feature maps [C, H, W] or [B, C, H, W]. |
| `Sigmoid(Tensor<>)` | Applies element-wise sigmoid to a tensor: 1 / (1 + exp(-x)). |
| `SoftmaxAlongClassDim(Tensor<>)` | Computes softmax along the class dimension, producing per-pixel probabilities. |
| `TextToWeights(String,Int32)` | Generates deterministic channel weights from text using character-level hashing. |
| `ThresholdMask(Tensor<>,Double)` | Thresholds a tensor to produce a binary {0, 1} mask. |
| `WarpMasksByAffinity(Tensor<>,Tensor<>)` | Warps masks from a reference frame to a target frame using pixel-wise affinity scores. |
| `WeightedChannelSum(Tensor<>,Double[])` | Computes weighted sum across the channel dimension. |

