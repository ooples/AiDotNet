---
title: "PanopticSegmentationBase<T>"
description: "Abstract base class for panoptic segmentation models that unify semantic and instance segmentation."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for panoptic segmentation models that unify semantic and instance segmentation.

## For Beginners

Panoptic segmentation gives you the most complete picture of a scene by
combining semantic segmentation (labeling regions like "road", "sky") with instance segmentation
(distinguishing individual objects like "car #1", "car #2").

Models extending this base class: Mask2Former, kMaX-DeepLab, OneFormer, ODISE.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PanopticSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,Int32)` | Initializes the base in native (trainable) mode. |
| `PanopticSegmentationBase(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumStuffClasses` |  |
| `NumThingClasses` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentPanoptic(Tensor<>)` |  |

