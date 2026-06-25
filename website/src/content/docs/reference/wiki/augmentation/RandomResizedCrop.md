---
title: "RandomResizedCrop<T>"
description: "Randomly crops and resizes a region of the image (PyTorch-style RandomResizedCrop)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Randomly crops and resizes a region of the image (PyTorch-style RandomResizedCrop).

## For Beginners

This picks a random-sized chunk of the image (from 8% to 100%
of the area) with a random shape (from slightly tall to slightly wide), then resizes it to
a fixed size. This teaches your model to recognize objects at different scales and crops.

## How It Works

RandomResizedCrop first extracts a random crop with an area between `MinScale`
and `MaxScale` of the original, with an aspect ratio between `MinRatio`
and `MaxRatio`, then resizes to the target output size. This is the standard
training augmentation for ImageNet and many other tasks.

**When to use:**

- Standard ImageNet training pipeline (the single most important augmentation)
- Any classification task as default training augmentation
- Self-supervised learning (SimCLR, BYOL, etc.)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomResizedCrop(Int32,Double,Double,Double,Double,InterpolationMode,Double)` | Creates a square RandomResizedCrop. |
| `RandomResizedCrop(Int32,Int32,Double,Double,Double,Double,InterpolationMode,Double)` | Creates a new RandomResizedCrop. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Interpolation` | Gets the interpolation mode. |
| `MaxRatio` | Gets the maximum aspect ratio (width/height). |
| `MaxScale` | Gets the maximum scale (fraction of image area). |
| `MinRatio` | Gets the minimum aspect ratio (width/height). |
| `MinScale` | Gets the minimum scale (fraction of image area). |
| `OutputHeight` | Gets the target output height. |
| `OutputWidth` | Gets the target output width. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` |  |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` |  |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` |  |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` |  |

