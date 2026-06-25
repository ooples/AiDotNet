---
title: "KOSMOS1<T>"
description: "KOSMOS-1: multimodal large language model with visual tokens embedded in causal LM."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

KOSMOS-1: multimodal large language model with visual tokens embedded in causal LM.

## For Beginners

KOSMOS-1 from Microsoft embeds image features directly into
a causal language model as if they were text tokens. Image patches from a CLIP ViT are
linearly projected into the same embedding space as text, then the combined image-text
sequence is processed by a standard causal transformer for unified multimodal understanding
and generation. Default values follow the original paper settings.

## How It Works

KOSMOS-1 (Huang et al., 2023) is a multimodal large language model that embeds visual tokens
directly into a causal language model. Image features from a CLIP ViT are linearly projected
into the same embedding space as text tokens, then the combined sequence is processed by a
causal transformer decoder for unified multimodal understanding and generation.

**References:**

- Paper: "Language Is Not All You Need: Aligning Perception with Language Models" (Huang et al., 2023)

**Architecture layout:** Vision encoder + projection live in
`Layers`; the causal transformer decoder lives in a private
auxiliary stream. `Predict` returns the vision-only embedding;
`String)` walks both streams to generate the multimodal output.

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using KOSMOS-1's unified multimodal causal LM architecture. |
| `GetExtraTrainableLayers` |  |

