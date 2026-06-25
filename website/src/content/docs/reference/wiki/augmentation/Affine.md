---
title: "Affine<T>"
description: "Applies random affine transformations (rotation, scale, shear, translation) to an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies random affine transformations (rotation, scale, shear, translation) to an image.

## For Beginners

Think of this as combining multiple geometric operations into one.
The image can be rotated, stretched, tilted (sheared), and moved around - all at once.
This creates more diverse training examples than applying each transformation separately.

## How It Works

Affine transformation is a general geometric transformation that preserves lines and
parallelism. It combines rotation, scaling, shearing, and translation in a single operation,
providing a powerful way to augment images with realistic geometric variations.

**When to use:**

- General image classification with geometric invariance
- Object detection where objects may be viewed from different angles
- When you need combined geometric variations efficiently

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Affine(Nullable<ValueTuple<Double,Double>>,Nullable<ValueTuple<Double,Double>>,Nullable<ValueTuple<Double,Double>>,Nullable<ValueTuple<Double,Double>>,Double,InterpolationMode,BorderMode,)` | Creates a new affine transformation augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BorderMode` | Gets the border mode when pixels fall outside the original image bounds. |
| `BorderValue` | Gets the constant value used when BorderMode is Constant. |
| `Interpolation` | Gets the interpolation mode for pixel sampling. |
| `RotationRange` | Gets the rotation angle range in degrees. |
| `ScaleRange` | Gets the scale factor range. |
| `ShearRange` | Gets the shear angle range in degrees. |
| `TranslationRange` | Gets the translation range as a fraction of image dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the affine transformation and returns transform parameters. |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after affine transformation. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after affine transformation. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after affine transformation. |

