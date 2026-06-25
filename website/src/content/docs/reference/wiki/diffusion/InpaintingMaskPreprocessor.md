---
title: "InpaintingMaskPreprocessor<T>"
description: "Inpainting mask preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Inpainting mask preprocessor for ControlNet conditioning.

## For Beginners

This prepares a mask that tells the AI which parts
of the image to regenerate (white) and which to keep (black). The feathering
option creates smooth transitions at mask edges for more natural blending.

## How It Works

Processes inpainting masks by binarizing and optionally feathering edges.
The output is a single-channel mask where 1.0 indicates regions to inpaint
and 0.0 indicates regions to preserve.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InpaintingMaskPreprocessor(Double,Int32)` | Initializes a new inpainting mask preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

