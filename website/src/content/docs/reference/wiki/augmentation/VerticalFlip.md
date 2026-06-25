---
title: "VerticalFlip<T>"
description: "Flips an image vertically (top-bottom mirror)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Flips an image vertically (top-bottom mirror).

## For Beginners

Think of this like flipping a photo upside down. The top of
the image becomes the bottom, and vice versa. Use this carefully because many objects
look unnatural when flipped vertically (imagine an upside-down car or person).

## How It Works

Vertical flipping mirrors the image along its horizontal center axis, swapping the top
and bottom portions. This is less commonly used than horizontal flipping because
most real-world objects have a consistent "up" orientation (gravity matters!).

**When to use:**

- Satellite or aerial imagery where "up" is arbitrary
- Microscopy images with no inherent orientation
- Abstract pattern recognition
- Medical imaging where orientation varies by acquisition

**When NOT to use:**

- Natural photography with gravity-oriented subjects
- Facial recognition or human pose estimation
- Vehicle recognition (cars don't drive upside down)
- Any task where vertical orientation is meaningful

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VerticalFlip(Double)` | Creates a new vertical flip augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the vertical flip transformation and returns transform parameters. |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after vertical flip. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after vertical flip. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after vertical flip. |

