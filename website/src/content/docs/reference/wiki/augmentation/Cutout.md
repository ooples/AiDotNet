---
title: "Cutout<T>"
description: "Randomly masks out (cuts out) rectangular regions of an image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Randomly masks out (cuts out) rectangular regions of an image.

## For Beginners

Imagine covering parts of a photo with sticky notes.
If you can still recognize what's in the photo with pieces hidden, you understand
the whole object better, not just one specific feature. Cutout does this automatically
during training, teaching the model to recognize objects even when parts are obscured.

## How It Works

Cutout is a regularization technique that randomly removes rectangular patches from
training images by filling them with a constant value (usually gray or black). This
forces the model to focus on multiple parts of the object rather than relying on
a single distinctive feature, improving robustness and generalization.

**When to use:**

- Image classification where objects might be partially occluded
- When you want to prevent the model from overfitting to specific features
- As a regularization technique to improve generalization

**When NOT to use:**

- Object detection or segmentation (might remove the entire object)
- When fine-grained features are crucial for classification
- Very small images where cutout would remove too much information

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Cutout(Int32,Int32,Int32,Int32,Int32,,Double)` | Creates a new cutout augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the fill value for the cutout regions. |
| `MaxHoleHeight` | Gets the maximum height of each hole. |
| `MaxHoleWidth` | Gets the maximum width of each hole. |
| `MinHoleHeight` | Gets the minimum height of each hole. |
| `MinHoleWidth` | Gets the minimum width of each hole. |
| `NumberOfHoles` | Gets the number of rectangular holes to cut out. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the cutout to the image. |
| `GetParameters` |  |

