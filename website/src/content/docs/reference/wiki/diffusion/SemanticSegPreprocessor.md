---
title: "SemanticSegPreprocessor<T>"
description: "Semantic segmentation preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Semantic segmentation preprocessor for ControlNet conditioning.

## For Beginners

This labels every pixel in the image (sky, person, car, etc.)
and paints them different colors. ControlNet uses this to generate images where
objects are in the same regions.

## How It Works

Produces semantic segmentation maps where each pixel is assigned a class label
encoded as a color. The output guides ControlNet to respect object boundaries and regions.

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

