---
title: "Gemma3<T>"
description: "Gemma 3: Google's 3B-72B VLM with Native Dynamic Resolution ViT and 128k context."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Gemma 3: Google's 3B-72B VLM with Native Dynamic Resolution ViT and 128k context.

## For Beginners

Gemma 3 from Google is a family of open vision-language models
ranging from 3 billion to 72 billion parameters. It can process images at their native
resolution using Dynamic Resolution ViT, meaning it adapts to each image's actual size
rather than forcing all images to a fixed size. With a 128K token context window, it can
handle very long conversations and documents. It supports 29 languages and uses a
SigLIP-based vision encoder with an MLP projection to connect visual features to the
language model. Default values follow the original paper settings.

## How It Works

Gemma 3 (Google, 2025) is a family of vision-language models ranging from 3B to 72B parameters.
It features Native Dynamic Resolution ViT for flexible image processing, 128k token context
window, and support for 29 languages. Uses SigLIP-based vision encoder with MLP projection.

**References:**

- Paper: "Gemma 3 Technical Report" (Google, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Gemma 3's Native Dynamic Resolution architecture. |

