---
title: "ScribblePreprocessor<T>"
description: "Scribble/sketch preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Scribble/sketch preprocessor for ControlNet conditioning.

## For Beginners

This turns your image into a rough sketch, like someone
quickly drew it with a pen. Unlike line art, scribbles are simpler and less
detailed, giving ControlNet more creative freedom.

## How It Works

Converts images into simplified scribble-like sketches by applying thresholded
edge detection with line thinning. Produces binary (black/white) output similar
to hand-drawn scribbles.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScribblePreprocessor(Double)` | Initializes a new scribble preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

