---
title: "NormalMapPreprocessor<T>"
description: "Normal map estimation preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Normal map estimation preprocessor for ControlNet conditioning.

## For Beginners

A normal map shows which direction each surface in the image
is facing. The R/G/B channels encode the X/Y/Z direction. This helps ControlNet
generate images with correct lighting and surface detail.

## How It Works

Estimates surface normals from image gradients, producing a 3-channel normal map
where RGB channels represent the X, Y, Z components of surface normals.

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

