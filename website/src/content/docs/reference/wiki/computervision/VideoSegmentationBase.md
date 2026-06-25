---
title: "VideoSegmentationBase<T>"
description: "Abstract base class for video segmentation models that track and segment objects across frames."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for video segmentation models that track and segment objects across frames.

## For Beginners

Video segmentation extends image segmentation to temporal sequences.
The key challenge is maintaining consistent object identity across frames — tracking objects
as they move, get occluded, or change appearance.

Models extending this base class: SAM 2, Cutie, XMem, DEVA, EfficientTAM, UniVS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32)` | Initializes the base in native (trainable) mode. |
| `VideoSegmentationBase(NeuralNetworkArchitecture<>,String,Int32,Int32)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTrackedObjects` |  |
| `SupportsStreaming` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCorrection(Int32,Tensor<>)` |  |
| `InitializeTracking(Tensor<>,Tensor<>,Int32[])` |  |
| `InitializeTrackingInternal(Tensor<>,Tensor<>,Int32[])` | Model-specific tracking initialization with first-frame features and masks. |
| `PropagateToFrame(Tensor<>)` |  |
| `PropagateToFrameInternal(Tensor<>,Int32)` | Model-specific mask propagation to the next frame. |
| `ResetTracking` |  |
| `ResetTrackingInternal` | Model-specific cleanup of tracking memory and state. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentFrameIndex` | Current frame index in the video sequence. |
| `_trackedObjectIds` | Object IDs currently being tracked. |
| `_trackingInitialized` | Whether tracking has been initialized with first-frame masks. |

