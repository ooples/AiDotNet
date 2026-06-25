---
title: "SpatialAugmentationBase<T, TData>"
description: "Base class for augmentations that transform spatial targets (bounding boxes, keypoints, masks)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation`

Base class for augmentations that transform spatial targets (bounding boxes, keypoints, masks).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpatialAugmentationBase(Double)` | Initializes a new spatial augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsBoundingBoxes` | Gets whether this augmentation supports bounding box transformation. |
| `SupportsKeypoints` | Gets whether this augmentation supports keypoint transformation. |
| `SupportsMasks` | Gets whether this augmentation supports mask transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(,AugmentationContext<>)` | Default implementation that calls ApplyWithTransformParams. |
| `ApplyWithTargets(AugmentedSample<,>,AugmentationContext<>)` | Applies the augmentation to data and all spatial targets. |
| `ApplyWithTransformParams(,AugmentationContext<>)` | Applies the augmentation and returns both the result and transform parameters. |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box according to the spatial transformation. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint according to the spatial transformation. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask according to the spatial transformation. |

