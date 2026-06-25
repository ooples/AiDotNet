---
title: "Rotation<T>"
description: "Rotates an image by a random angle within a specified range."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Rotates an image by a random angle within a specified range.

## For Beginners

Imagine tilting your camera slightly when taking a photo.
The same object photographed at a slight angle is still the same object. This augmentation
teaches your model to recognize objects even when they're not perfectly aligned.

## How It Works

Rotation randomly rotates the image around its center point by an angle sampled
from the specified range. This simulates viewing objects from slightly different
angles, which helps the model become robust to orientation variations.

**When to use:**

- Object classification where objects might appear at different angles
- Document analysis (scanned documents may be slightly tilted)
- Medical imaging where acquisition angle varies
- Satellite imagery where orientation is arbitrary

**When NOT to use:**

- Facial recognition (faces should be upright)
- Handwriting recognition (letters need consistent orientation)
- Tasks where specific orientation is part of the classification

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Rotation(Double,Double,Double,BorderMode,,InterpolationMode)` | Creates a new rotation augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BorderMode` | Gets the border fill mode when pixels fall outside the original image bounds. |
| `BorderValue` | Gets the constant value used when BorderMode is Constant. |
| `Interpolation` | Gets the interpolation mode for pixel sampling. |
| `MaxAngle` | Gets the maximum rotation angle in degrees. |
| `MinAngle` | Gets the minimum rotation angle in degrees (can be negative for counter-clockwise). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the rotation transformation and returns transform parameters. |
| `GetParameters` |  |
| `SamplePixel(ImageTensor<>,Double,Double,Int32)` | Samples a pixel value at non-integer coordinates using the configured interpolation. |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after rotation. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after rotation. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after rotation. |

