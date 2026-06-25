---
title: "Emu2<T>"
description: "Emu2: scaled 37B unified understanding and generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

Emu2: scaled 37B unified understanding and generation model.

## For Beginners

Emu2 scales the original Emu architecture to 37 billion parameters,
using a larger EVA-CLIP-E vision encoder and a more powerful LLaMA-based decoder. It excels
at in-context learning for multimodal tasks — given a few examples of image-text pairs, it
can generalize to new tasks without fine-tuning. Default values follow the original paper
settings.

## How It Works

Emu2 (Sun et al., 2024) scales the Emu architecture to 37B parameters by using a larger
EVA-CLIP-E encoder and a more powerful LLaMA-based decoder. It demonstrates strong in-context
learning for multimodal tasks and can generate both text and images from interleaved inputs.

**References:**

- Paper: "Generative Multimodal Models are In-Context Learners" (Sun et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates using Emu2's scaled 37B unified architecture. |
| `GetExtraTrainableLayers` |  |

