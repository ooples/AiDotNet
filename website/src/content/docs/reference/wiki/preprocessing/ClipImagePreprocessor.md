---
title: "ClipImagePreprocessor<T>"
description: "Preprocesses images for CLIP (Contrastive Language-Image Pre-training) models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Image`

Preprocesses images for CLIP (Contrastive Language-Image Pre-training) models.

## For Beginners

Before CLIP can "see" an image, it needs to be prepared:

1. **Resize**: Images come in all sizes (1000x2000, 50x50, etc.)

CLIP expects a specific size (like 224x224 pixels).

2. **Normalize**: Pixel values (0-255) are scaled and shifted using

standard values from ImageNet dataset. This helps the model work consistently.

3. **Format**: The image is arranged as [R, G, B] channels first,

then height and width. This is called "channels-first" format.

Example:

- Original: 1920x1080 photo with RGB values 0-255
- After preprocessing: 224x224 tensor with normalized values around [-2, 2]

## How It Works

CLIP models expect images to be preprocessed in a specific way:

1. Resize to a square size (typically 224x224 or 336x336)
2. Normalize pixel values using ImageNet mean and standard deviation
3. Convert to tensor format [channels, height, width]

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClipImagePreprocessor(Int32,[],[])` | Initializes a new instance of the ClipImagePreprocessor class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ImageSize` | Gets the target image size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BilinearInterpolate(Tensor<>,Int32,,,Int32,Int32)` | Performs bilinear interpolation at a fractional position. |
| `ExpandGrayscale(Tensor<>)` | Expands a grayscale image to 3 channels by repeating. |
| `ExpandSingleChannel(Tensor<>)` | Expands a single-channel image to 3 channels. |
| `ExtractFirstImage(Tensor<>)` | Extracts the first image from a batch. |
| `FindMax(Tensor<>)` | Finds the maximum value in a tensor. |
| `NormalizeFormat(Tensor<>)` | Normalizes the tensor format to channels-first [C, H, W]. |
| `NormalizePixels(Tensor<>)` | Normalizes pixel values using ImageNet statistics. |
| `Preprocess(Tensor<>)` | Preprocesses an image for CLIP input. |
| `PreprocessBatch(IEnumerable<Tensor<>>)` | Preprocesses a batch of images for CLIP input. |
| `Resize(Tensor<>,Int32,Int32)` | Resizes an image using bilinear interpolation. |
| `TakeFirstChannels(Tensor<>,Int32)` | Takes only the first N channels from an image. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_engine` | The computational engine for tensor operations. |
| `_imageSize` | The target image size (height and width). |
| `_mean` | The normalization mean values for RGB channels (ImageNet standard). |
| `_numOps` | The numeric operations helper for type T. |
| `_std` | The normalization standard deviation values for RGB channels (ImageNet standard). |

