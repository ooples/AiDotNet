---
title: "PixtralLarge<T>"
description: "Pixtral Large: Mistral's 124B decoder + 1B vision encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Pixtral Large: Mistral's 124B decoder + 1B vision encoder.

## For Beginners

Pixtral Large is the scaled-up version of Pixtral, increasing
from 12B to 124B total parameters with a 1B vision encoder (up from 400M). It uses
Mistral-Large as the language backbone, which provides much stronger reasoning,
instruction following, and knowledge capabilities than the standard Mistral model.
The larger vision encoder also captures more detailed visual information. This model
targets server-side deployment where maximum quality is more important than model
size, making it suitable for high-stakes applications like medical image analysis,
detailed document understanding, and complex visual reasoning. Default values follow
the original paper settings.

## How It Works

Pixtral Large (Mistral, 2024) scales up the Pixtral architecture to 124B parameters with
a 1B vision encoder. It uses the Mistral-Large language backbone for improved reasoning and
instruction following with high-resolution image understanding.

**References:**

- Paper: "Pixtral Large" (Mistral, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from an image using Pixtral Large's variable-resolution architecture. |

