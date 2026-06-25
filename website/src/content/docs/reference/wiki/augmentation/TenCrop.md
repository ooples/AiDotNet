---
title: "TenCrop<T>"
description: "Extracts ten crops from an image: five crops plus their horizontal flips."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Extracts ten crops from an image: five crops plus their horizontal flips.

## For Beginners

This creates 10 different views of your image: 5 crops
(4 corners + center) and their mirror images. By averaging your model's predictions
over all 10 views, you can get more reliable results at test time.

## How It Works

TenCrop extends `FiveCrop` by also flipping each of the five crops
horizontally, yielding 10 total views. This provides more diverse test-time augmentation
for improved classification accuracy.

**When to use:**

- Test-time augmentation when accuracy is paramount
- Competition settings where every fraction of a percent matters

**When NOT to use:**

- Training (use RandomCrop + RandomHorizontalFlip instead)
- Real-time applications (10x computation overhead)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TenCrop(Int32,Boolean,Double)` | Creates a new TenCrop augmentation with square crops. |
| `TenCrop(Int32,Int32,Boolean,Double)` | Creates a new TenCrop augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CropHeight` | Gets the height of each crop. |
| `CropWidth` | Gets the width of each crop. |
| `UseVerticalFlip` | Gets whether to use vertical flips instead of horizontal flips. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the augmentation. |
| `GetCrops(ImageTensor<>)` | Extracts the ten crops from the image. |
| `GetParameters` |  |

