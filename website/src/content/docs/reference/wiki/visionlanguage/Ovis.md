---
title: "Ovis<T>"
description: "Ovis: structural embedding alignment VLM with visual token scaling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Ovis: structural embedding alignment VLM with visual token scaling.

## For Beginners

Ovis takes a different approach to connecting vision and language
than most models. Instead of just projecting visual features through an MLP, it uses
"structural embedding alignment" — a technique that ensures the visual tokens not only
have the right dimensions for the language model, but also preserve the structural
relationships between them (like spatial layout and relative importance). With visual
token scaling, it can adjust how many visual tokens to use based on the complexity of
the image, using fewer tokens for simple images and more for complex ones. Default
values follow the original paper settings.

## How It Works

Ovis (Lu et al., 2024) introduces structural embedding alignment, a technique that aligns the
structural properties of visual embeddings with the text embedding space of the language model.
Rather than using a simple MLP projection, Ovis learns a visual token scaling approach that
maps visual features into the language model's embedding space while preserving their structural
relationships. This produces better-aligned representations for the language model to work with.

**References:**

- Paper: "Ovis: Structural Embedding Alignment for Multimodal Large Language Model" (2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Ovis's visual embedding table for structural alignment. |

