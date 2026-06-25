---
title: "Pad<T>"
description: "Pads an image with configurable padding amounts and fill modes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Pads an image with configurable padding amounts and fill modes.

## For Beginners

Padding makes an image larger by adding extra pixels around
the edges. This is useful when you need a specific size but don't want to crop or stretch
the image. Black padding (constant=0) is most common.

## How It Works

Padding adds pixels around the border of an image. Multiple fill modes control
how the new pixels are filled:

- **Constant**: Fills with a fixed value (e.g., 0 for black, 255 for white)
- **Edge**: Extends the edge pixel values outward
- **Reflect**: Mirrors the image at the boundary
- **Symmetric**: Like reflect but includes the boundary pixel

**When to use:**

- Making images a specific size without distortion
- Preparing for convolutions that reduce spatial size
- Adding borders for visual presentation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Pad(Int32,Int32,Int32,Int32,PaddingMode,Double,Double)` | Creates a new pad augmentation with individual padding per side. |
| `Pad(Int32,Int32,PaddingMode,Double,Double)` | Creates a new pad augmentation with separate horizontal and vertical padding. |
| `Pad(Int32,PaddingMode,Double,Double)` | Creates a new pad augmentation with uniform padding. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the constant fill value (used when Mode is Constant). |
| `Mode` | Gets the padding mode. |
| `PadBottom` | Gets the padding on the bottom side. |
| `PadLeft` | Gets the padding on the left side. |
| `PadRight` | Gets the padding on the right side. |
| `PadTop` | Gets the padding on the top side. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the padding operation and returns transformation parameters. |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after padding. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after padding. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after padding. |

