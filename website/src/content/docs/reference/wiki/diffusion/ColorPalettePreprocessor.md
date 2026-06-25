---
title: "ColorPalettePreprocessor<T>"
description: "Color palette extraction preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Color palette extraction preprocessor for ControlNet conditioning.

## For Beginners

This reduces your image to a small number of main colors,
like a paint-by-numbers version. ControlNet uses this to generate new images
that match the color scheme of your original.

## How It Works

Extracts dominant colors from an image and produces a quantized color map.
Each pixel is mapped to the nearest dominant color, creating a simplified
color palette representation for color-guided generation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColorPalettePreprocessor(Int32)` | Initializes a new color palette preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

