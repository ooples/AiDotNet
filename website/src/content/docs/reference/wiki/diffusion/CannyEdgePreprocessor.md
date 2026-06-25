---
title: "CannyEdgePreprocessor<T>"
description: "Canny edge detection preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Canny edge detection preprocessor for ControlNet conditioning.

## For Beginners

This finds the outlines/edges in your image.
The result looks like a drawing showing only the borders of objects.
ControlNet uses this to generate new images with the same structure.

## How It Works

Applies the Canny edge detection algorithm to extract edges from images.
The output is a single-channel binary edge map used for structural control.

Reference: Canny, "A Computational Approach to Edge Detection", IEEE TPAMI 1986

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CannyEdgePreprocessor(Double,Double)` | Initializes a new Canny edge preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

