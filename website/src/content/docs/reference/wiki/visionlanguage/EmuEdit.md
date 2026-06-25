---
title: "EmuEdit<T>"
description: "Emu Edit: precise image editing via recognition and generation tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Editing`

Emu Edit: precise image editing via recognition and generation tasks.

## For Beginners

Emu Edit is a vision-language model for instruction-based image
editing that understands natural language edit commands. Default values follow the original
paper settings.

## How It Works

Emu Edit (Meta, 2024) enables precise image editing by jointly training on recognition and
generation tasks. The model learns to understand editing instructions through multi-task learning
on 16 editing tasks including region-based editing, color/texture modifications, and object
addition/removal, using a diffusion-based generation backbone conditioned on text instructions.

**References:**

- Paper: "Emu Edit: Precise Image Editing via Recognition and Generation Tasks" (Meta, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `EditImage(Tensor<>,String)` | Edits an image using Emu Edit's recognition-guided precise editing pipeline. |

