---
title: "Molmo<T>"
description: "Molmo: open VLM with pointing and grounding capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Molmo: open VLM with pointing and grounding capabilities.

## For Beginners

Molmo from AI2 is notable for being fully open — not just
the model weights, but also the training data (called PixMo) is publicly available,
which is rare for high-performing multimodal models. Beyond standard visual QA, Molmo
can "point" to objects in images by outputting coordinate positions, making it useful
for tasks like "where is the cat in this image?" where it can indicate the exact location.
It achieves state-of-the-art performance among open models while being completely
transparent in how it was trained. Default values follow the original paper settings.

## How It Works

Molmo (AI2, 2024) from the Allen Institute for AI is a fully open multimodal model with both
open weights and open training data (PixMo). It features pointing and grounding capabilities —
the model can point to specific objects in images by generating (x, y) coordinates and can
localize objects described in natural language. Molmo achieves state-of-the-art performance
among open models while providing complete transparency in its training data and methodology.

**References:**

- Paper: "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models" (AI2, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Molmo's pointing-capable architecture with diverse data mixture. |

