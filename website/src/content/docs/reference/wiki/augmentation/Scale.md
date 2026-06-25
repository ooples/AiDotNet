---
title: "Scale<T>"
description: "Scales an image by a random factor within a specified range."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Scales an image by a random factor within a specified range.

## For Beginners

Think of this like zooming in or out on your camera.
The same object photographed from closer or farther appears at different sizes.
This augmentation teaches your model to recognize objects regardless of their size.

## How It Works

Scale randomly resizes the image by a factor sampled from the specified range.
This simulates viewing objects from different distances, helping the model
become robust to scale variations.

**When to use:**

- Object detection where objects appear at various sizes
- Image classification with variable-sized subjects
- When training data lacks size diversity

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Scale(Double,Double,Double,InterpolationMode,BorderMode,)` | Creates a new scale augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BorderMode` | Gets the border mode when pixels fall outside the original image bounds. |
| `BorderValue` | Gets the constant value used when BorderMode is Constant. |
| `Interpolation` | Gets the interpolation mode for pixel sampling. |
| `MaxScale` | Gets the maximum scale factor. |
| `MinScale` | Gets the minimum scale factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the scale transformation and returns transform parameters. |
| `GetParameters` |  |
| `SamplePixel(ImageTensor<>,Double,Double,Int32)` | Samples a pixel value at non-integer coordinates. |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after scaling. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after scaling. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after scaling. |

