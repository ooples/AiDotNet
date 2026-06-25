---
title: "Phi3Vision<T>"
description: "Phi-3-Vision: compact 4.2B parameter model with strong vision capabilities via curated data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Phi-3-Vision: compact 4.2B parameter model with strong vision capabilities via curated data.

## For Beginners

Phi-3-Vision from Microsoft proves that you don't need a
massive model to get strong visual AI. At just 4.2 billion parameters, it's small enough
to run on a phone, yet achieves performance that rivals much larger models. The secret
is high-quality training data — Microsoft carefully curated the training examples to
maximize what the model learns from each one. It uses a CLIP-ViT vision encoder connected
to the Phi-3 language model through an MLP projection, keeping the architecture simple
while relying on data quality for performance. Default values follow the original paper
settings.

## How It Works

Phi-3-Vision (Abdin et al., 2024) is a compact multimodal model with 4.2B parameters that
achieves strong performance through high-quality curated training data rather than model scale.
It uses CLIP-ViT with MLP projection into the Phi-3 language model backbone.

**References:**

- Paper: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone" (Abdin et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Phi-3-Vision's compact data-centric architecture. |

