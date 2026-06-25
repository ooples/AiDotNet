---
title: "IDEFICS2<T>"
description: "IDEFICS2: 8B efficient VLM with SigLIP encoder and Mistral-7B decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

IDEFICS2: 8B efficient VLM with SigLIP encoder and Mistral-7B decoder.

## For Beginners

IDEFICS2 is a much more efficient 8 billion parameter successor
to the original 80B IDEFICS. It replaces the OpenCLIP encoder with SigLIP and uses Mistral-7B
as the language backbone, introducing learned perceiver pooling and native resolution image
processing that splits images into sub-images for better document understanding. Default
values follow the original paper settings.

## How It Works

IDEFICS2 (Laurencon et al., 2024) is an 8B parameter efficient VLM that replaces the
OpenCLIP encoder with SigLIP and uses Mistral-7B as the language backbone. It introduces
a learned perceiver pooling strategy and native resolution image processing via sub-image
splitting for improved document understanding.

**References:**

- Paper: "What matters when building vision-language models?" (Laurencon et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using IDEFICS2's efficient 8B SigLIP+Mistral architecture. |
| `GetExtraTrainableLayers` |  |

