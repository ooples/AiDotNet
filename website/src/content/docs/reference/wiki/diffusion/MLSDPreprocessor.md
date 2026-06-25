---
title: "MLSDPreprocessor<T>"
description: "MLSD (Mobile Line Segment Detection) preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

MLSD (Mobile Line Segment Detection) preprocessor for ControlNet conditioning.

## For Beginners

This finds straight lines in your image (walls, edges of buildings,
table edges). It's especially useful for architectural images where you want to
preserve geometric structure.

## How It Works

Detects straight line segments in images, useful for architectural and geometric
structure preservation. The output shows detected line segments on a black background.

Reference: Gu et al., "Towards Light-weight and Real-time Line Segment Detection", AAAI 2022

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

