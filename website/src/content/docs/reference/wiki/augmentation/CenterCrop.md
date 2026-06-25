---
title: "CenterCrop<T>"
description: "Crops the center region of an image to a specified size."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Crops the center region of an image to a specified size.

## For Beginners

Think of this like taking a photo and cutting out the middle
rectangle. The center of the image usually contains the most important content, so this
is a reliable way to get a fixed-size input for your model during testing.

## How It Works

Center cropping extracts a fixed-size region from the center of the image. This is commonly
used during evaluation/inference to ensure consistent framing, and as part of the standard
ImageNet preprocessing pipeline (resize to 256, then center crop to 224).

**When to use:**

- Evaluation/inference preprocessing (standard for ImageNet models)
- When you need deterministic cropping (no randomness)
- As part of a resize-then-crop pipeline

**When NOT to use:**

- Training (use RandomCrop instead for data diversity)
- When important content is near the edges

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CenterCrop(Int32,Double)` | Creates a new center crop augmentation with square output. |
| `CenterCrop(Int32,Int32,Double)` | Creates a new center crop augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CropHeight` | Gets the output height after cropping. |
| `CropWidth` | Gets the output width after cropping. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the center crop and returns transformation parameters. |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after center crop. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after center crop. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after center crop. |

