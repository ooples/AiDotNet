---
title: "CutMix<T>"
description: "Cuts a rectangular region from one image and pastes it onto another (CutMix augmentation)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Cuts a rectangular region from one image and pastes it onto another (CutMix augmentation).

## For Beginners

Imagine cutting a square from one photo and pasting it onto
another, like cutting a face from one picture and pasting it onto a landscape. The label
becomes a mix based on how much of each image is visible. Unlike MixUp which creates
"ghostly" blends, CutMix keeps sharp boundaries which can be more natural-looking.

## How It Works

CutMix is a regularization technique that combines aspects of Cutout and MixUp. It cuts
a rectangular region from one training image and pastes it onto another image. The labels
are mixed proportionally to the area of the cut region:
y' = λy₁ + (1-λ)y₂
where λ is the ratio of the original image area that remains.

**When to use:**

- Image classification as a stronger regularizer than Cutout
- When you want benefits of both Cutout (occlusion) and MixUp (label smoothing)
- Often combined with MixUp in training pipelines

**When NOT to use:**

- Object detection (the cut region might remove important objects entirely)
- Semantic segmentation (hard to handle cut region masks)
- When precise localization information is important

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CutMix(Double,Double,Double,Double)` | Creates a new CutMix augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxCutRatio` | Gets the maximum ratio of the image area that should be cut. |
| `MinCutRatio` | Gets the minimum ratio of the image area that should be cut. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies CutMix augmentation (single image version - requires external pairing). |
| `ApplyCutMix(ImageTensor<>,ImageTensor<>,Vector<>,Vector<>,AugmentationContext<>)` | Applies CutMix by cutting a region from image2 and pasting it onto image1. |
| `GetParameters` |  |

