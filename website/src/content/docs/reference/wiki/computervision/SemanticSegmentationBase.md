---
title: "SemanticSegmentationBase<T>"
description: "Abstract base class for semantic segmentation models that assign a class label to every pixel."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for semantic segmentation models that assign a class label to every pixel.

## For Beginners

Semantic segmentation answers "what is this pixel?" for every pixel in an
image. This base class provides the shared infrastructure for models like SegFormer, SegNeXt,
InternImage, and DiffSeg — all of which produce a per-pixel class label map.

Extending this class gives you:

- Dual-mode support (native training + ONNX inference)
- Automatic class map extraction (argmax of logits)
- Probability map generation (softmax of logits)
- All common serialization and batch handling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SemanticSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32)` | Initializes the base in native (trainable) mode. |
| `SemanticSegmentationBase(NeuralNetworkArchitecture<>,String,Int32)` | Initializes the base in ONNX (inference-only) mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ArgmaxAlongClassDim(Tensor<>)` | Computes argmax along the class dimension (dim 0 for [C, H, W] or dim 1 for [B, C, H, W]). |
| `GetClassMap(Tensor<>)` |  |
| `GetProbabilityMap(Tensor<>)` |  |
| `SoftmaxAlongClassDim(Tensor<>)` | Computes softmax along the class dimension. |

