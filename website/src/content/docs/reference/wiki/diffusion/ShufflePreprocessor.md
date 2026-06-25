---
title: "ShufflePreprocessor<T>"
description: "Shuffle preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Shuffle preprocessor for ControlNet conditioning.

## For Beginners

This cuts your image into small squares and randomly rearranges
them, like a jigsaw puzzle that's been mixed up. ControlNet uses this to transfer
the colors and textures of your image without copying its exact layout.

## How It Works

Shuffles image patches to create a permuted version of the input. This preserves
overall color distribution and texture while destroying spatial structure, enabling
color/style transfer without structural copying.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShufflePreprocessor(Int32,Int32)` | Initializes a new shuffle preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

