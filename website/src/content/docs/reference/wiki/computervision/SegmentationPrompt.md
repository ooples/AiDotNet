---
title: "SegmentationPrompt<T>"
description: "Represents a user prompt for interactive/promptable segmentation models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Common`

Represents a user prompt for interactive/promptable segmentation models.

## For Beginners

A prompt tells the model what you want to segment.
You can use different types of prompts:

- Points: Click on the object (foreground) or background
- Boxes: Draw a rectangle around the object
- Masks: Provide a rough outline for refinement
- Text: Describe what to segment in natural language

You can combine multiple prompt types for better results.

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioPrompt` | Audio prompt for multi-modal models (e.g., SEEM). |
| `Boxes` | Box prompts as [N, 4] (x1, y1, x2, y2 per box in pixel coordinates). |
| `MaskInput` | Mask prompt [H, W] where positive values indicate foreground. |
| `NegativeTextPrompt` | Negative text prompt describing what to exclude. |
| `PointLabels` | Point labels: 1 = foreground, 0 = background, -1 = ambiguous. |
| `Points` | Point prompts as [N, 2] (x, y pairs in pixel coordinates). |
| `PrimaryPromptType` | Type of the primary prompt being used. |
| `ReferenceImage` | Reference image for in-context learning (e.g., SegGPT). |
| `ReferenceMask` | Reference mask corresponding to the reference image. |
| `ReturnLowResLogits` | Whether to return low-resolution logits for iterative refinement. |
| `ReturnMultipleMasks` | Whether to return multiple mask proposals (SAM-style) or a single best mask. |
| `TargetObjectIds` | Target object IDs for tracking correction in video segmentation. |
| `TextPrompt` | Text prompt for language-guided segmentation. |

