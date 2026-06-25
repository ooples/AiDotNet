---
title: "PromptableSegmentationBase<T>"
description: "Abstract base class for promptable segmentation models like SAM that accept user prompts (points, boxes, masks) to segment specific objects."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for promptable segmentation models like SAM that accept user prompts
(points, boxes, masks) to segment specific objects.

## For Beginners

Promptable segmentation lets you point at, draw a box around, or
describe what you want to segment. The model first encodes the image (which is expensive but
done once), then quickly processes each prompt against the encoded representation.

Models extending this base class: SAM, SAM 2, SAM-HQ, SegGPT, SEEM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptableSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32)` | Initializes the base in native (trainable) mode. |
| `PromptableSegmentationBase(NeuralNetworkArchitecture<>,String,Int32)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsBoxPrompts` |  |
| `SupportsMaskPrompts` |  |
| `SupportsPointPrompts` |  |
| `SupportsTextPrompts` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EncodeImage(Tensor<>)` | Encodes an image into an embedding for subsequent prompted segmentation. |
| `EnsureImageSet` | Ensures an image has been set before prompting. |
| `SegmentEverything` |  |
| `SegmentFromBox(Tensor<>)` |  |
| `SegmentFromMask(Tensor<>)` |  |
| `SegmentFromPoints(Tensor<>,Tensor<>)` |  |
| `SetImage(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_imageEmbedding` | Cached image embedding from the most recent SetImage call. |
| `_imageSet` | Whether an image has been encoded and is ready for prompting. |

