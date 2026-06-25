---
title: "OutpaintingMaskGenerator<T>"
description: "Generates masks for outpainting by marking regions outside the original image bounds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Generates masks for outpainting by marking regions outside the original image bounds.

## For Beginners

Outpainting extends an image beyond its borders. This utility
creates the mask that tells the model which parts are the original image (don't change)
and which parts need to be generated (extend). The feathering option makes the
transition smoother.

## How It Works

Creates a mask where the original image region is unmasked (0) and extended regions
are masked (1). Supports extending in any direction with optional feathering at
the boundary for smooth blending.

When used as an `IDataTransformer`, the Transform method
takes the original image tensor and generates an outpainting mask based on the padding
configured in the constructor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OutpaintingMaskGenerator(Int32,Int32,Int32,Int32,Int32)` | Initializes a new instance of the `OutpaintingMaskGenerator` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ColumnIndices` |  |
| `IsFitted` |  |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Tensor<>)` |  |
| `FitTransform(Tensor<>)` |  |
| `Generate(Int32,Int32,Int32,Int32,Int32,Int32)` | Generates an outpainting mask for the given canvas and image bounds. |
| `GetFeatureNamesOut(String[])` |  |
| `InverseTransform(Tensor<>)` |  |
| `Transform(Tensor<>)` | Generates an outpainting mask from the input image tensor's dimensions. |

