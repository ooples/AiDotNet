---
title: "ISpatialAugmentation<T, TData>"
description: "Interface for augmentations that can transform spatial targets (bounding boxes, keypoints, segmentation masks)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Interface for augmentations that can transform spatial targets
(bounding boxes, keypoints, segmentation masks).

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsBoundingBoxes` | Gets whether this augmentation supports bounding box transformation. |
| `SupportsKeypoints` | Gets whether this augmentation supports keypoint transformation. |
| `SupportsMasks` | Gets whether this augmentation supports segmentation mask transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTargets(AugmentedSample<,>,AugmentationContext<>)` | Applies the augmentation and transforms all spatial targets. |

