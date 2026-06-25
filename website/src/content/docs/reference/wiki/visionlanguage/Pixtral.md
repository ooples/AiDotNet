---
title: "Pixtral<T>"
description: "Pixtral: Mistral's 12B decoder + 400M vision encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Pixtral: Mistral's 12B decoder + 400M vision encoder.

## For Beginners

Pixtral is Mistral's entry into multimodal AI, pairing a compact
400M parameter vision encoder with the powerful 12B Mistral language model. It processes
high-resolution images at 1024 pixels and uses an MLP projection to connect visual features
to the language decoder. The result is a model that combines Mistral's strong language
capabilities (reasoning, coding, instruction following) with image understanding. It can
describe images, answer visual questions, read text in photos, and reason about visual
content. Default values follow the original paper settings.

## How It Works

Pixtral (Mistral, 2024) is a 12B parameter multimodal model combining a 400M parameter
vision encoder with the Mistral language model. It processes high-resolution images at 1024px
and uses MLP projection for vision-language alignment.

**References:**

- Paper: "Pixtral 12B" (Mistral, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Pixtral's jointly-trained vision encoder architecture. |

