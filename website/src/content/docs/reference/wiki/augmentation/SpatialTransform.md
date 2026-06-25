---
title: "SpatialTransform<T>"
description: "Applies spatial transformations (flips, rotations) consistently to all frames."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Video`

Applies spatial transformations (flips, rotations) consistently to all frames.

## For Beginners

Spatial transform applies image augmentations like
horizontal flip or rotation to all frames in the video consistently.
This ensures the transformation is coherent across the entire video.

## How It Works

**Available transforms:**

- Horizontal flip (mirror left-right)
- Vertical flip (mirror top-bottom)
- 90° rotation (clockwise)
- 180° rotation
- 270° rotation (counter-clockwise)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpatialTransform(Double,Double,Double,Double)` | Creates a new spatial transform augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableHorizontalFlip` | Gets or sets whether to enable horizontal flip. |
| `EnableRotation90` | Gets or sets whether to enable 90° rotations. |
| `EnableVerticalFlip` | Gets or sets whether to enable vertical flip. |
| `HorizontalFlipProbability` | Gets the probability of horizontal flip (when enabled). |
| `VerticalFlipProbability` | Gets the probability of vertical flip (when enabled). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>[],AugmentationContext<>)` |  |
| `GetParameters` |  |

