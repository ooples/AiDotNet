---
title: "RgbToBgr<T>"
description: "Converts between RGB and BGR color channel orderings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Converts between RGB and BGR color channel orderings.

## For Beginners

Different libraries store color channels in different orders.
Most deep learning frameworks use RGB, but OpenCV uses BGR. If you loaded an image with
OpenCV and want to use it with a PyTorch model, you need to swap the channel order.
This transform handles that conversion in both directions.

## How It Works

RGB and BGR differ only in the order of color channels. RGB (Red, Green, Blue) is the
standard for most frameworks (PyTorch, TensorFlow), while BGR (Blue, Green, Red) is
used by OpenCV. This transform swaps the first and third channels.

**When to use:**

- When loading images with OpenCV (BGR) for use with PyTorch/TF models (RGB)
- When preprocessing for models trained with BGR input (some older Caffe models)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RgbToBgr(Double)` | Creates a new RGB/BGR channel swap. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(ImageTensor<>,AugmentationContext<>)` | Swaps the R and B channels. |
| `GetParameters` |  |

