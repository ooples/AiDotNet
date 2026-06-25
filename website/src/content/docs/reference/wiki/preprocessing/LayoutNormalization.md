---
title: "LayoutNormalization<T>"
description: "LayoutNormalization - Document layout normalization utilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Document`

LayoutNormalization - Document layout normalization utilities.

## For Beginners

Different documents have different sizes and proportions.
This tool standardizes them for AI models:

- Resize to target dimensions
- Preserve aspect ratio options
- Handle padding and cropping
- Normalize orientation

Key features:

- Multiple normalization strategies
- Aspect ratio preservation
- Smart padding and cropping
- Batch processing support

Example usage:

## How It Works

LayoutNormalization provides utilities for normalizing document layouts
to standard sizes and aspect ratios for consistent model input.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayoutNormalization` | Creates a new LayoutNormalization instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CenterCrop(Tensor<>,Int32,Int32)` | Crops the center of the image to target dimensions. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the layout normalization utility. |
| `Flip(Tensor<>,Boolean)` | Flips the image horizontally or vertically. |
| `GetAspectRatio(Tensor<>)` | Computes aspect ratio of the image. |
| `NormalizeOrientation(Tensor<>,Boolean)` | Detects and corrects document orientation (portrait vs landscape). |
| `Process(Tensor<>,Int32,Int32,NormalizationStrategy)` | Normalizes a document image to the specified dimensions. |
| `ResizeAndCrop(Tensor<>,Int32,Int32)` | Resizes to cover target dimensions then crops to exact size. |
| `ResizeWithPadding(Tensor<>,Int32,Int32,)` | Resizes the image preserving aspect ratio and adds padding. |
| `Rotate90(Tensor<>,Boolean)` | Rotates the image 90 degrees. |
| `Stretch(Tensor<>,Int32,Int32)` | Stretches the image to target dimensions (may distort aspect ratio). |

