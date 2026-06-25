---
title: "PathVLM<T>"
description: "PathVLM: histopathology-specific vision-language model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Medical`

PathVLM: histopathology-specific vision-language model.

## For Beginners

PathVLM is a vision-language model specialized for computational
pathology and histopathology image analysis. Default values follow the original paper
settings.

## How It Works

PathVLM (2024) is a histopathology-specific vision-language model that processes whole-slide
images at multiple magnification levels. It uses multi-scale patch processing to capture both
cellular-level details and tissue-level context, enabling pathology report generation,
diagnosis assistance, and visual question answering on histopathology specimens.

**References:**

- Paper: "PathVLM: A Vision-Language Model for Computational Pathology (Various, 2024)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a histopathology image using PathVLM's multi-scale encoding. |

