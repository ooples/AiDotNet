---
title: "ImageTensor<T>"
description: "Represents an image as a tensor with image-specific metadata and operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Represents an image as a tensor with image-specific metadata and operations.

## For Beginners

An image on a computer is stored as numbers representing
pixel colors. This class represents those numbers in a way that's optimized for
machine learning, while keeping track of important details like whether the image
is in RGB or BGR format.

## How It Works

ImageTensor wraps a Tensor<T> to provide image-specific functionality:

- Channel ordering (CHW vs HWC)
- Color space awareness (RGB, BGR, HSV, etc.)
- Normalization state tracking
- Image-specific operations (crop, resize, color conversion)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageTensor(Int32,Int32,Int32,ChannelOrder,ColorSpace)` | Creates an ImageTensor with specified dimensions. |
| `ImageTensor(Int32,Int32,Int32,Int32,ChannelOrder,ColorSpace)` | Creates a batched ImageTensor with specified dimensions. |
| `ImageTensor(Tensor<>,ChannelOrder,ColorSpace)` | Creates an ImageTensor from an existing tensor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets the batch size (1 for single images). |
| `ChannelOrder` | Gets or sets the channel ordering. |
| `Channels` | Gets the number of channels. |
| `ColorSpace` | Gets or sets the color space. |
| `Data` | Gets the underlying tensor data. |
| `Height` | Gets the image height in pixels. |
| `IsNormalized` | Gets or sets whether the image is normalized to [0, 1]. |
| `Metadata` | Gets or sets additional metadata. |
| `NormalizationMean` | Gets or sets the normalization mean (per channel). |
| `NormalizationStd` | Gets or sets the normalization std (per channel). |
| `OriginalRange` | Gets or sets the original value range before normalization. |
| `Width` | Gets the image width in pixels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateIndex(Int32,Int32,Int32,Int32)` | Calculates the flat index for a pixel location. |
| `CalculateIndexStatic(Int32,Int32,Int32,Int32,Int32,Int32,Int32,ChannelOrder)` | Static version of index calculation for transposition. |
| `Clone` | Creates a deep copy of this image tensor. |
| `Crop(Int32,Int32,Int32,Int32)` | Extracts a rectangular region from the image. |
| `GetDimensions` | Gets the tensor dimensions as an array. |
| `GetPixel(Int32,Int32,Int32)` | Gets a pixel value at the specified coordinates. |
| `GetPixelChannels(Int32,Int32)` | Gets all channel values at a pixel location. |
| `ParseDimensions` | Parses dimensions from the tensor based on channel order. |
| `SetPixel(Int32,Int32,Int32,)` | Sets a pixel value at the specified coordinates. |
| `SetPixelChannels(Int32,Int32,[])` | Sets all channel values at a pixel location. |
| `ToChannelOrder(ChannelOrder)` | Converts this image to a different channel order. |
| `TransposeData(Tensor<>,Tensor<>,ChannelOrder,ChannelOrder)` | Transposes data between channel orderings. |

