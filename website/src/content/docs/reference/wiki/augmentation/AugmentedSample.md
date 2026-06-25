---
title: "AugmentedSample<T, TData>"
description: "Represents a sample with its data and associated spatial targets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Represents a sample with its data and associated spatial targets.

## For Beginners

When you rotate an image, you also need to rotate
any bounding boxes, keypoints, or segmentation masks associated with it.
This class keeps all these elements together so they transform correctly.

## How It Works

An augmented sample bundles together the data (e.g., image) with all its
associated annotations that need to be transformed together when spatial
augmentations are applied.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentedSample()` | Creates a new augmented sample with only data. |
| `AugmentedSample(,Vector<>)` | Creates a new augmented sample with data and labels. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundingBoxes` | Gets or sets the bounding boxes for object detection. |
| `Data` | Gets or sets the primary data (e.g., image). |
| `HasBoundingBoxes` | Gets whether this sample has any bounding boxes. |
| `HasKeypoints` | Gets whether this sample has any keypoints. |
| `HasMasks` | Gets whether this sample has any segmentation masks. |
| `HasSpatialTargets` | Gets whether this sample has any spatial targets. |
| `Keypoints` | Gets or sets the keypoints for pose estimation. |
| `Labels` | Gets or sets the label(s) for this sample. |
| `Masks` | Gets or sets the segmentation masks. |
| `Metadata` | Gets or sets additional metadata for this sample. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of this sample including all targets. |

