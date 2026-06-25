---
title: "Janus<T>"
description: "Janus: decoupled visual encoding for understanding vs generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

Janus: decoupled visual encoding for understanding vs generation.

## For Beginners

Janus is a unified vision-language model from DeepSeek that
uses separate visual paths for understanding and generation. Default values follow the
original paper settings.

## How It Works

Janus (DeepSeek, 2024) decouples visual encoding into separate pathways for understanding
and generation. It uses a SigLIP encoder for visual understanding tasks and a VQ tokenizer
for image generation, both feeding into a shared LLM backbone. This decoupling resolves
the tension between high-level semantic features needed for understanding and low-level
visual tokens needed for generation.

**References:**

- Paper: "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation" (DeepSeek, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using Janus's decoupled understanding encoder. |
| `GenerateImage(String)` | Generates an image from text using Janus's decoupled generation encoder. |

