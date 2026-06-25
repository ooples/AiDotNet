---
title: "IPromptableSegmentation<T>"
description: "Interface for interactive, promptable segmentation models like SAM that accept user prompts (points, boxes, masks, text) to segment specific objects."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for interactive, promptable segmentation models like SAM that accept
user prompts (points, boxes, masks, text) to segment specific objects.

## For Beginners

Promptable segmentation is like pointing at something in a photo
and having the model outline it for you.

Prompt types:

- Points: Click on an object and the model segments it
- Boxes: Draw a rectangle around an object for more precise segmentation
- Masks: Provide a rough mask and the model refines it
- Text: Describe what to segment (for models that support it)

Models implementing this interface:

- SAM / SAM 2 (Meta, foundation model for segmentation)
- SAM-HQ (high-quality boundaries)
- SegGPT (in-context learning)
- SEEM (multi-modal prompts including audio)

## How It Works

Promptable segmentation models first encode an image, then accept various types of prompts
to segment specific regions. This two-stage design allows efficient interactive use where
the image is encoded once and multiple prompts can be processed quickly.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsBoxPrompts` | Gets whether this model supports box prompts. |
| `SupportsMaskPrompts` | Gets whether this model supports mask prompts. |
| `SupportsPointPrompts` | Gets whether this model supports point prompts. |
| `SupportsTextPrompts` | Gets whether this model supports text prompts. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentEverything` | Generates masks for the entire image without any prompts (automatic mode). |
| `SegmentFromBox(Tensor<>)` | Segments a region indicated by a bounding box prompt. |
| `SegmentFromMask(Tensor<>)` | Segments a region using a rough mask as a prompt. |
| `SegmentFromPoints(Tensor<>,Tensor<>)` | Segments a region indicated by point prompts. |
| `SetImage(Tensor<>)` | Encodes an image for subsequent prompted segmentation. |

