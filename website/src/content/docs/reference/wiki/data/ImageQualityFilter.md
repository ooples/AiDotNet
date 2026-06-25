---
title: "ImageQualityFilter"
description: "Filters images based on resolution, aspect ratio, and pixel statistics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Filters images based on resolution, aspect ratio, and pixel statistics.

## How It Works

Checks image metadata and pixel-level statistics to detect low-quality images:
blank/solid-color images, extreme aspect ratios, tiny resolutions, and corrupt files.
Works on raw pixel data represented as flattened arrays.

## Methods

| Method | Summary |
|:-----|:--------|
| `FilterByDimensions(IReadOnlyList<Int32>,IReadOnlyList<Int32>)` | Filters images by dimension, returning indices of images that should be removed. |
| `PassesDimensionCheck(Int32,Int32)` | Checks whether an image passes quality filters based on dimensions. |
| `PassesFileSizeCheck(Int64)` | Checks whether a file size meets minimum requirements. |
| `PassesPixelCheck(Double[])` | Checks whether pixel values indicate a quality image (not blank, not solid). |

