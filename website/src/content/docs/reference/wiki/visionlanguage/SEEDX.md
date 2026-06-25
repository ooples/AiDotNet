---
title: "SEEDX<T>"
description: "SEED-X: multi-granularity comprehension and generation model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Unified`

SEED-X: multi-granularity comprehension and generation model.

## For Beginners

SEED-X is a unified model for multi-granularity visual
understanding and generation from Tencent. Default values follow the original paper
settings.

## How It Works

SEED-X (Tencent, 2024) is a multimodal model with unified multi-granularity comprehension
and generation. It processes visual information at multiple granularity levels from global
scene understanding to fine-grained detail recognition, using a SEED visual tokenizer that
captures both high-level semantics and low-level visual details for generation tasks.

**References:**

- Paper: "SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation" (Tencent, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from image using SEED-X's multi-granularity visual comprehension. |
| `GenerateImage(String)` | Generates an image from text using SEED-X's multi-granularity generation pipeline. |

