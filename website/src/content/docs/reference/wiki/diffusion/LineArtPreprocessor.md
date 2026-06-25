---
title: "LineArtPreprocessor<T>"
description: "Line art extraction preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

Line art extraction preprocessor for ControlNet conditioning.

## For Beginners

This turns your photo into a clean line drawing, like a coloring
book page. It's different from edge detection because it focuses on artistic lines
rather than just boundaries.

## How It Works

Extracts clean line art from images, producing a single-channel sketch-like output.
Unlike edge detection, line art preserves artistic line quality and thickness variation.

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

