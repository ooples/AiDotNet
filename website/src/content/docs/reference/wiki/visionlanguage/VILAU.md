---
title: "VILAU<T>"
description: "VILA-U: unified VLM with autoregressive visual generation and understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

VILA-U: unified VLM with autoregressive visual generation and understanding.

## For Beginners

VILA-U extends VILA to not just understand images but also
generate them. Most vision-language models are one-way — they look at images and produce
text. VILA-U works both ways: it can analyze an image and describe it, or take a text
description and generate an image. It does this by converting images into discrete tokens
(like a visual vocabulary) that the language model can both read and write. This unified
approach means a single model handles image understanding, text generation, and image
generation without needing separate specialized models for each task. Default values
follow the original paper settings.

## How It Works

VILA-U (NVIDIA, 2024) unifies visual understanding and visual generation in a single
autoregressive model. Unlike most VLMs that can only understand images and generate text,
VILA-U can also generate images autoregressively by predicting visual tokens one at a time.
It uses a discrete visual tokenizer to convert images into a sequence of tokens that the
language model can both consume (for understanding) and produce (for generation), creating
a truly unified visual-linguistic model.

**References:**

- Paper: "VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation" (NVIDIA, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using VILA-U's unified understanding and generation architecture. |

