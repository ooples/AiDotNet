---
title: "SmartEdit<T>"
description: "SmartEdit: enhanced instruction understanding for complex image editing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Editing`

SmartEdit: enhanced instruction understanding for complex image editing.

## For Beginners

SmartEdit is a vision-language model for complex instruction-based
image editing that handles nuanced and multi-step editing commands. Default values follow
the original paper settings.

## How It Works

SmartEdit (2024) explores complex instruction-based image editing by leveraging multimodal
LLMs to understand nuanced editing instructions. It enhances instruction comprehension through
bidirectional interaction between the understanding and generation components, enabling edits
that require reasoning about spatial relationships, object attributes, and complex scene context.

**References:**

- Paper: "SmartEdit: Exploring Complex Instruction-based Image Editing with Multimodal LLMs" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `EditImage(Tensor<>,String)` | Edits an image using SmartEdit's bidirectional MLLM-diffusion interaction. |

