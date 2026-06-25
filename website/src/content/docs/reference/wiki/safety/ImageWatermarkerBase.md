---
title: "ImageWatermarkerBase<T>"
description: "Abstract base class for image watermarking modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Watermarking`

Abstract base class for image watermarking modules.

## For Beginners

This base class provides common code for all image watermarkers.
Each watermarker type extends this and adds its own way of embedding invisible
signatures in images.

## How It Works

Provides shared infrastructure for image watermarkers including strength
configuration and frequency domain utilities. Concrete implementations provide
the actual watermarking technique (frequency, neural, invisible spatial).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageWatermarkerBase(Double)` | Initializes the image watermarker base. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `WatermarkStrength` | The watermark strength factor (0.0 to 1.0). |

