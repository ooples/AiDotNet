---
title: "Perspective<T>"
description: "Applies a random perspective transformation to an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies a random perspective transformation to an image.

## For Beginners

Imagine tilting a photo in 3D space — objects closer to you
appear larger, and objects farther away appear smaller. This augmentation simulates that
effect, teaching your model to recognize objects regardless of viewing angle.

## How It Works

Perspective transformation simulates viewing the image from different angles by applying
a projective (homography) transform. Each corner of the image is displaced by a random
amount, creating a realistic 3D perspective effect.

**When to use:**

- Document recognition (photos of documents taken at angles)
- Street sign/license plate recognition
- Any task where the camera angle may vary

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Perspective(Double,InterpolationMode,Double,Double)` | Creates a new perspective transformation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistortionScale` | Gets the maximum distortion scale (fraction of image size). |
| `FillValue` | Gets the fill value for areas outside the transformed image. |
| `Interpolation` | Gets the interpolation mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies perspective transformation and returns transform parameters. |
| `ComputePerspectiveMatrix(Double[][],Double[][])` | Computes a 3x3 perspective transformation matrix using 4 point correspondences. |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` |  |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` |  |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` |  |

