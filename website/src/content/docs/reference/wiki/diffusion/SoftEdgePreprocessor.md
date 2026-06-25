---
title: "SoftEdgePreprocessor<T>"
description: "Soft edge detection preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Soft edge detection preprocessor for ControlNet conditioning.

## For Beginners

Unlike Canny edges which are sharp black-and-white lines,
soft edges have smooth gradients. This gives ControlNet more flexibility in how
strictly it follows the structure.

## How It Works

Produces soft (non-binary) edge maps with smooth transitions, providing more
flexible structural guidance than hard Canny edges.

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

