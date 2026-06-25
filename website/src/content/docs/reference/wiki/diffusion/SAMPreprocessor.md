---
title: "SAMPreprocessor<T>"
description: "SAM (Segment Anything Model) preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

SAM (Segment Anything Model) preprocessor for ControlNet conditioning.

## For Beginners

SAM can segment any object in an image. This preprocessor
creates a colored map where each object gets its own color, helping ControlNet
understand object boundaries for generation.

## How It Works

Produces segmentation masks using gradient-based region detection as an
approximation of SAM-style segmentation. Each detected region receives
a unique color label in the output.

Reference: Kirillov et al., "Segment Anything", ICCV 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAMPreprocessor(Double)` | Initializes a new SAM preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

