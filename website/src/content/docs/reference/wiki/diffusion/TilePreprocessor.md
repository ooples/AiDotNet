---
title: "TilePreprocessor<T>"
description: "Tile preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Tile preprocessor for ControlNet conditioning.

## For Beginners

This creates a blurry version of your image that
ControlNet uses to keep the same colors and general layout, while letting
the AI add sharp details. It's commonly used for upscaling and detail enhancement.

## How It Works

Produces a blurred/downsampled version of the input image for tile-based
ControlNet conditioning. This guides the model to preserve overall color
and composition while allowing fine detail regeneration.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TilePreprocessor(Int32)` | Initializes a new tile preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

