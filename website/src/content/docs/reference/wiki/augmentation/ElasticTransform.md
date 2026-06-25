---
title: "ElasticTransform<T>"
description: "Applies elastic deformation to an image (Simard et al., 2003)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies elastic deformation to an image (Simard et al., 2003).

## For Beginners

Imagine placing your image on a rubber sheet and randomly
pushing and pulling different parts. The result looks naturally distorted, like how
handwriting varies slightly each time you write the same letter. This is one of the most
effective augmentations for digit/character recognition.

## How It Works

Elastic deformation generates random displacement fields, smooths them with a Gaussian
filter, and applies them to warp the image. This creates realistic local distortions
similar to handwriting variations or biological tissue deformation.

**When to use:**

- Handwritten digit/character recognition (MNIST, EMNIST)
- Medical image segmentation (tissue deformation)
- Any task where local shape variations are natural

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticTransform(Double,Double,Double,Double)` | Creates a new elastic transform. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha parameter controlling displacement magnitude. |
| `FillValue` | Gets the fill value for out-of-bounds pixels. |
| `Sigma` | Gets the sigma parameter controlling smoothness of the displacement field. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Applies the elastic deformation. |
| `GetParameters` |  |

