---
title: "FrequencyImageWatermarker<T>"
description: "Image watermarker that embeds watermarks in the frequency domain using DCT/DWT coefficients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Image watermarker that embeds watermarks in the frequency domain using DCT/DWT coefficients.

## For Beginners

This watermarker hides a signature in the image's frequency
components — the mathematical representation of patterns and textures. The watermark
survives common image operations like saving as JPEG because it's embedded in the
most robust part of the frequency spectrum.

## How It Works

Embeds watermark bits by modifying mid-frequency DCT coefficients. Mid-frequencies are
chosen because they survive JPEG compression while remaining imperceptible. Detection
extracts the same coefficients and checks for the embedded pattern.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FrequencyImageWatermarker(Double)` | Initializes a new frequency-domain image watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Tensor<>)` |  |
| `EvaluateImage(Tensor<>)` |  |

