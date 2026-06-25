---
title: "RandomCrop<T>"
description: "Randomly crops a region from the image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Randomly crops a region from the image.

## For Beginners

Think of this like taking a photo with a camera that randomly
zooms in on different parts of the scene. Even if only part of an object is visible,
it's still that object. This teaches your model to recognize objects even when they're
partially out of frame.

## How It Works

Random cropping extracts a random rectangular region from the image. This is one of the
most effective augmentations for teaching models to recognize objects even when partially
visible or at different positions within the frame.

**When to use:**

- Image classification where objects may be partially visible
- Training for translation invariance
- When input images are larger than needed for the model

**When NOT to use:**

- Object detection (may crop out target objects entirely)
- When the full context of the image is important

**Padding behavior:** When `UseScaleCropping` is false and the input
image is smaller than the requested crop dimensions (`CropWidth` x `CropHeight`),
the output tensor will be zero-padded in the areas beyond the source image bounds. If you require
the source image to be at least as large as the crop dimensions, validate this before applying
the augmentation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomCrop(Int32,Int32,Double)` | Creates a new random crop augmentation with fixed output size. |
| `RandomCrop(Int32,Int32,Double,Double,Double,Double,Double)` | Creates a new random resized crop augmentation (scale-based cropping). |

## Properties

| Property | Summary |
|:-----|:--------|
| `CropHeight` | Gets the output height after cropping. |
| `CropWidth` | Gets the output width after cropping. |
| `MaxAspectRatio` | Gets the maximum aspect ratio (width/height) for the crop. |
| `MaxScale` | Gets the maximum scale factor for the crop region relative to image size. |
| `MinAspectRatio` | Gets the minimum aspect ratio (width/height) for the crop. |
| `MinScale` | Gets the minimum scale factor for the crop region relative to image size. |
| `UseScaleCropping` | Gets whether to use scale-based random cropping (like RandomResizedCrop). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the random crop transformation and returns transform parameters. |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after random crop. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after random crop. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after random crop. |

