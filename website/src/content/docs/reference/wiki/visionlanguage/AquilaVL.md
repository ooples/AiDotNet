---
title: "AquilaVL<T>"
description: "AquilaVL: bilingual VLM aligned with Chinese and English visual-language understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

AquilaVL: bilingual VLM aligned with Chinese and English visual-language understanding.

## For Beginners

AquilaVL is a bilingual (Chinese/English) model that can look
at an image and answer questions about it in either language. It uses a SigLIP vision encoder
to understand what is in the image, projects those visual features through an MLP into the
Qwen2.5 language model, and generates text responses. It handles images at their native
resolution by splitting them into tiles, so it can understand both the big picture and fine
details. Default values follow the original paper settings.

## How It Works

AquilaVL (BAAI, 2024) is a bilingual vision-language model designed for strong performance
in both Chinese and English visual understanding. It follows a LLaVA-style architecture using
a SigLIP vision encoder to extract dense visual features, a 2-layer MLP projector with SiLU
activation to map visual tokens into the language model's embedding space, and a Qwen2.5
decoder backbone for text generation. AquilaVL supports dynamic resolution by tiling images
and encoding each tile independently, enabling high-fidelity understanding of detailed scenes.

**References:**

- Paper: "AquilaVL: Advanced Vision-Language Model" (BAAI, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Aquila-VL's SigLIP + Qwen2.5 architecture. |

