---
title: "ImageAugmenterBase<T>"
description: "Base class for image data augmentations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Augmentation.Image`

Base class for image data augmentations.

## For Beginners

Image augmentation transforms images to improve model
robustness to variations in viewpoint, lighting, and appearance. Common techniques include:

- Geometric transforms: flips, rotations, scaling, cropping
- Color transforms: brightness, contrast, saturation, hue
- Noise and blur: Gaussian noise, blur, sharpening
- Regularization: cutout, mixup, cutmix

## How It Works

Image data is represented as an ImageTensor with dimensions (height, width, channels).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageAugmenterBase(Double)` | Initializes a new image augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetChannels(ImageTensor<>)` | Gets the number of channels in the image. |
| `GetHeight(ImageTensor<>)` | Gets the height of the image. |
| `GetWidth(ImageTensor<>)` | Gets the width of the image. |
| `HasAlpha(ImageTensor<>)` | Checks if the image has an alpha channel. |
| `IsGrayscale(ImageTensor<>)` | Checks if the image is grayscale (single channel). |

