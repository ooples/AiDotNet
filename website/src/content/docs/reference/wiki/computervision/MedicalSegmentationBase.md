---
title: "MedicalSegmentationBase<T>"
description: "Abstract base class for medical image segmentation models handling 2D slices and 3D volumes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for medical image segmentation models handling 2D slices and 3D volumes.

## For Beginners

Medical segmentation helps doctors by automatically outlining organs,
tumors, and other structures in medical images. These models handle special requirements
like 3D volumetric processing (CT/MRI scans are stacks of slices), multi-modal imaging,
and the very high accuracy needed for clinical use.

Models extending this base class: nnU-Net, TransUNet, Swin-UNETR, MedSAM, MedSAM 2, MedNeXt.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedicalSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,IEnumerable<String>)` | Initializes the base in native (trainable) mode. |
| `MedicalSegmentationBase(NeuralNetworkArchitecture<>,String,Int32,IEnumerable<String>)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedModalities` |  |
| `Supports2D` |  |
| `Supports3D` |  |
| `SupportsFewShot` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentFewShot(Tensor<>,Tensor<>,Tensor<>)` |  |
| `SegmentFewShotInternal(Tensor<>,Tensor<>,Tensor<>)` | Model-specific few-shot segmentation. |
| `SegmentSlice(Tensor<>)` |  |
| `SegmentVolume(Tensor<>)` |  |
| `SlidingWindowInference(Tensor<>,ValueTuple<Int32,Int32,Int32>,Double)` | Applies sliding window inference for 3D volumes that exceed GPU memory. |

