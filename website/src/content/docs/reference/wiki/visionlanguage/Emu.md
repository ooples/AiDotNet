---
title: "Emu<T>"
description: "Emu: unified VQA, captioning, and image generation via EVA-CLIP + LLM + regression head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

Emu: unified VQA, captioning, and image generation via EVA-CLIP + LLM + regression head.

## For Beginners

Emu unifies visual understanding (answering questions about images,
generating captions) and image generation in a single model. It uses an EVA-CLIP vision encoder
with a LLaMA-based language model, and a visual regression head that can map LLM outputs back
to visual embeddings for image generation via diffusion. Default values follow the original
paper settings.

## How It Works

Emu (Sun et al., 2023) unifies visual understanding and generation in a single model. It uses
an EVA-CLIP vision encoder to extract visual features, a Causal Transformer (LLaMA-based) as
the multimodal LLM, and a visual regression head that maps LLM hidden states back to the
visual embedding space for image generation via a diffusion decoder.

**References:**

- Paper: "Generative Pretraining in Multimodality" (Sun et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates using Emu's unified understanding + generation architecture. |
| `GetExtraTrainableLayers` |  |

