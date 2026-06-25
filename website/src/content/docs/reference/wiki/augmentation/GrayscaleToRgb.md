---
title: "GrayscaleToRgb<T>"
description: "Converts a grayscale image to RGB by replicating the single channel across all three channels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts a grayscale image to RGB by replicating the single channel across all three channels.

## For Beginners

Some models require 3-channel (RGB) input. If you have a
grayscale image with only 1 channel, this copies that single channel three times to
create a compatible 3-channel image. The image still looks gray, but now has the right
number of channels.

## How It Works

This creates a 3-channel image from a single-channel grayscale image by duplicating the
luminance value into the R, G, and B channels. The resulting image will look identical
to the grayscale original but is compatible with models that expect 3-channel input.

**When to use:**

- When feeding grayscale images to models trained on RGB data
- When using pretrained models that expect 3-channel input
- Mixing grayscale and color datasets in the same pipeline

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GrayscaleToRgb(Double)` | Creates a new grayscale to RGB conversion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Converts the grayscale image to RGB. |
| `GetParameters` |  |

