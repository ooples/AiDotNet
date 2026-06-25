---
title: "ImageQualityFilterOptions"
description: "Configuration options for image quality filtering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for image quality filtering.

## How It Works

Filters images based on resolution, aspect ratio, and statistical quality metrics.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxAspectRatio` | Maximum allowed aspect ratio (width/height or height/width). |
| `MaxDominantColorRatio` | Maximum fraction of pixels with the same value (detects solid-color images). |
| `MinFileSize` | Minimum file size in bytes (filters corrupt/tiny files). |
| `MinHeight` | Minimum image height in pixels. |
| `MinPixelStdDev` | Minimum standard deviation of pixel values (detects blank images). |
| `MinUniqueColors` | Minimum number of unique pixel values. |
| `MinWidth` | Minimum image width in pixels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

