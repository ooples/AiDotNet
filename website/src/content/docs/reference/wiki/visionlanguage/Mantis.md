---
title: "Mantis<T>"
description: "Mantis: interleaved multi-image VLM for complex visual reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Mantis: interleaved multi-image VLM for complex visual reasoning.

## For Beginners

Most vision-language models process one image at a time, but
Mantis is designed to handle multiple images together. It can compare two photos, track
changes across a sequence of images, or reason about relationships between different
views. It does this through interleaved instruction tuning — training on examples where
multiple images and text are woven together, teaching the model to understand how images
relate to each other. This makes it great for tasks like "what changed between these two
photos?" or "describe the story across these images." Default values follow the original
paper settings.

## How It Works

Mantis (Jiang et al., 2024) is specifically designed for multi-image reasoning tasks where the
model must understand relationships between multiple images presented together. It uses interleaved
multi-image instruction tuning, where images and text are interleaved in the input sequence,
enabling the model to compare, contrast, and reason across multiple visual inputs simultaneously.
This is particularly useful for tasks like "spot the difference", visual storytelling across
image sequences, and multi-view reasoning.

**References:**

- Paper: "Mantis: Interleaved Multi-Image Instruction Tuning" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Mantis's multi-image interleaved attention architecture. |

