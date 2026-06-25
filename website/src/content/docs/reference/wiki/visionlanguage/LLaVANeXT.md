---
title: "LLaVANeXT<T>"
description: "LLaVA-NeXT: improved reasoning, OCR, and world knowledge via dynamic high-resolution AnyRes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

LLaVA-NeXT: improved reasoning, OCR, and world knowledge via dynamic high-resolution AnyRes.

## For Beginners

LLaVA-NeXT extends LLaVA-1.5 with a key innovation called
AnyRes (Any Resolution) — instead of resizing all images to a fixed size, it splits
high-resolution images into multiple tiles and processes each tile separately. This
dramatically improves the model's ability to read text in images (OCR), understand
charts and tables, and reason about fine visual details. It uses LLaMA-3 as the language
backbone for stronger reasoning. The result is a model much better at tasks requiring
visual precision, like reading small text or understanding complex diagrams. Default
values follow the original paper settings.

## How It Works

LLaVA-NeXT (Liu et al., 2024) extends LLaVA-1.5 with AnyRes dynamic resolution that processes
high-resolution images by splitting them into multiple tiles, significantly improving OCR, chart
understanding, and fine-grained visual reasoning while using LLaMA-3 as the language backbone.

**References:**

- Paper: "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge" (Liu et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using LLaVA-NeXT's AnyRes dynamic high-resolution architecture. |

