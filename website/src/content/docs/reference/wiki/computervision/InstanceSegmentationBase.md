---
title: "InstanceSegmentationBase<T>"
description: "Abstract base class for instance segmentation models that detect and mask individual object instances."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for instance segmentation models that detect and mask individual object instances.

## For Beginners

Instance segmentation finds each individual object in an image and provides
a pixel-level mask for each one. Unlike semantic segmentation (all cars are "car"), instance
segmentation distinguishes car #1 from car #2.

Models extending this base class: YOLOv9-Seg, YOLO11-Seg, YOLOv12-Seg, YOLO26-Seg, Mask2Former, MaskDINO.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstanceSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,Double,Double)` | Initializes the base in native (trainable) mode. |
| `InstanceSegmentationBase(NeuralNetworkArchitecture<>,String,Int32,Int32,Double,Double)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConfidenceThreshold` |  |
| `MaxInstances` |  |
| `NmsThreshold` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectInstances(Tensor<>)` |  |

