---
title: "ContentShufflePreprocessor<T>"
description: "Content shuffle preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Content shuffle preprocessor for ControlNet conditioning.

## For Beginners

This is like the Shuffle preprocessor but smarter —
it groups similar-looking parts of the image together before rearranging.
ControlNet uses this to capture the "feel" and colors of your image without
copying the exact arrangement.

## How It Works

Performs content-aware shuffling that rearranges image regions based on
similarity rather than random permutation. Groups similar pixels together
while destroying spatial layout, preserving texture and color distributions
more faithfully than random shuffling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContentShufflePreprocessor(Int32,Int32)` | Initializes a new content shuffle preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

