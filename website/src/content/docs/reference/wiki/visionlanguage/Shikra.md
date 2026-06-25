---
title: "Shikra<T>"
description: "Shikra: referential dialogue model with coordinate output for grounding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Grounding`

Shikra: referential dialogue model with coordinate output for grounding.

## For Beginners

Shikra is a vision-language model that grounds objects by generating
their coordinates as plain text numbers in natural dialogue. Default values follow the original
paper settings.

## How It Works

Shikra (Chen et al., 2023) is a referential dialogue model that treats spatial coordinates
as plain-text numbers in the LLM output vocabulary. Instead of using special tokens or
external detection heads, it generates normalized bounding box coordinates as decimal text
tokens via CLIP ViT visual encoding interleaved with instruction tokens, enabling natural
language grounding through straightforward autoregressive generation.

**References:**

- Paper: "Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic" (SenseTime, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GroundText(Tensor<>,String)` | Grounds text using Shikra's referential dialogue with numeric coordinate output. |

