---
title: "HorizontalFlip<T>"
description: "Flips an image horizontally (left-right mirror)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Flips an image horizontally (left-right mirror).

## For Beginners

Think of this like looking in a mirror. The left side of the
image becomes the right side, and vice versa. This is useful when training image classifiers
because a cat is still a cat whether it's facing left or right.

## How It Works

Horizontal flipping mirrors the image along its vertical center axis, swapping the left
and right sides. This is one of the most commonly used augmentations because many objects
look the same when flipped horizontally (e.g., cars, animals, faces in frontal view).

**When to use:**

- Image classification where horizontal orientation doesn't matter
- Object detection (boxes will be flipped automatically)
- Pose estimation (keypoints will be swapped correctly)

**When NOT to use:**

- Text recognition (text would become unreadable when mirrored)
- Reading direction matters (left-to-right vs right-to-left)
- Asymmetric objects where orientation is meaningful (e.g., traffic signs with arrows)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HorizontalFlip(Double)` | Creates a new horizontal flip augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWithTransformParams(ImageTensor<>,AugmentationContext<>)` | Applies the horizontal flip transformation and returns transform parameters. |
| `GetParameters` |  |
| `TransformBoundingBox(BoundingBox<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a bounding box after horizontal flip. |
| `TransformKeypoint(Keypoint<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a keypoint after horizontal flip. |
| `TransformMask(SegmentationMask<>,IDictionary<String,Object>,AugmentationContext<>)` | Transforms a segmentation mask after horizontal flip. |

