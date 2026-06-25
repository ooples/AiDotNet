---
title: "FiveCrop<T>"
description: "Extracts five crops from an image: four corners and center."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Extracts five crops from an image: four corners and center.

## For Beginners

Instead of just looking at the center of an image, this
creates 5 different views by cropping from each corner and the center. During testing,
you can run your model on all 5 crops and average the predictions for better accuracy.

## How It Works

FiveCrop produces five fixed crops of the specified size from one image: top-left,
top-right, bottom-left, bottom-right, and center. This is commonly used at test time
to improve accuracy by averaging predictions over multiple views of the same image.

**When to use:**

- Test-time augmentation (TTA) for improved classification accuracy
- When you want deterministic multi-crop evaluation

**When NOT to use:**

- Training (use RandomCrop instead)
- When inference speed is critical (5x the computation)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FiveCrop(Int32,Double)` | Creates a new FiveCrop augmentation with square crops. |
| `FiveCrop(Int32,Int32,Double)` | Creates a new FiveCrop augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CropHeight` | Gets the height of each crop. |
| `CropWidth` | Gets the width of each crop. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the augmentation. |
| `GetCrops(ImageTensor<>)` | Extracts the five crops from the image. |
| `GetParameters` |  |

