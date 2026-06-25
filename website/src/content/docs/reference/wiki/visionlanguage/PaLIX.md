---
title: "PaLIX<T>"
description: "PaLI-X: scaled PaLI to 55B with improved component fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

PaLI-X: scaled PaLI to 55B with improved component fine-tuning.

## For Beginners

PaLI-X scales the PaLI architecture to 55 billion parameters
by pairing the ViT-22B (the largest dense Vision Transformer) with a 32B UL2 language model.
It introduces component-wise fine-tuning that allows separately optimizing the vision and
language components for improved benchmark performance across captioning, VQA, and
document understanding tasks. Default values follow the original paper settings.

## How It Works

PaLI-X (Chen et al., 2023) scales PaLI to 55B parameters by using a ViT-22B vision encoder
and a 32B UL2 language model. It introduces component-wise fine-tuning for improved performance
on vision-language benchmarks.

**References:**

- Paper: "PaLI-X: On Scaling up a Multilingual Vision and Language Model" (Chen et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using PaLI-X's scaled ViT-22B + UL2 architecture. |

