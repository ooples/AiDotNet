---
title: "Maya<T>"
description: "Maya: multilingual multimodal VLM supporting 8 languages."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Maya: multilingual multimodal VLM supporting 8 languages.

## For Beginners

Maya is built specifically for multilingual use — it can look
at an image and answer questions about it in 8 different languages. While most vision-language
models work well only in English, Maya was instruction-tuned with visual QA data across
multiple languages so it can understand and respond in Hindi, Spanish, French, Arabic,
Bengali, Chinese, Japanese, and English. This makes it useful for applications serving
diverse global audiences. Default values follow the original paper settings.

## How It Works

Maya (Gupta et al., 2024) is a multilingual multimodal model designed to understand images
and generate text in 8 languages. Unlike most VLMs that focus primarily on English, Maya uses
instruction fine-tuning with multilingual visual question answering data to support diverse
languages including Hindi, Spanish, French, Arabic, Bengali, Chinese, Japanese, and English.
It follows the LLaVA-style architecture with a vision encoder and language model connected
through a projector.

**References:**

- Paper: "Maya: An Instruction Finetuned Multilingual Multimodal Model" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Maya's multilingual vision-language architecture. |

