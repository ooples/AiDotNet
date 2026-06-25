---
title: "MPLUGDocOwl<T>"
description: "mPLUG-DocOwl: modular MLLM for document understanding with visual abstractor."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Document`

mPLUG-DocOwl: modular MLLM for document understanding with visual abstractor.

## For Beginners

mPLUG-DocOwl is a document understanding model from Alibaba
with a visual abstractor for efficient document processing. Default values follow the
original paper settings.

## How It Works

mPLUG-DocOwl (Alibaba, 2023) is a modularized multimodal large language model for document
understanding. It uses a visual abstractor module to compress high-resolution document image
features into a compact set of visual tokens, bridging the ViT encoder and LLM decoder
for efficient document VQA, information extraction, and document classification.

**References:**

- Paper: "mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding" (Alibaba, 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a document image using mPLUG-DocOwl's visual abstractor pipeline. |

