---
title: "InvisibleImageWatermarker<T>"
description: "Image watermarker that embeds imperceptible spatial-domain watermarks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Image watermarker that embeds imperceptible spatial-domain watermarks.

## For Beginners

This watermarker hides a signature by making tiny changes
to pixel values — so small that the human eye cannot see them. It's like writing
a message in invisible ink within the image data.

## How It Works

Embeds watermark bits by making sub-pixel-level modifications to pixel values in
the spatial domain. Uses least-significant-bit (LSB) encoding with error diffusion
to spread the watermark energy and avoid visual artifacts.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InvisibleImageWatermarker(Double)` | Initializes a new invisible spatial-domain image watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Tensor<>)` |  |
| `EvaluateImage(Tensor<>)` |  |

