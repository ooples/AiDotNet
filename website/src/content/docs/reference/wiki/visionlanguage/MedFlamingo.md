---
title: "MedFlamingo<T>"
description: "Med-Flamingo: few-shot medical visual question answering via Flamingo architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Medical`

Med-Flamingo: few-shot medical visual question answering via Flamingo architecture.

## For Beginners

Med-Flamingo is a vision-language model for few-shot medical
visual question answering using the Flamingo architecture. Default values follow the
original paper settings.

## How It Works

Med-Flamingo (2023) adapts the OpenFlamingo architecture for few-shot medical visual question
answering. It uses gated cross-attention layers to interleave medical image features with
text tokens and a perceiver resampler to compress visual information, enabling the model
to learn from just a few medical image-text examples without extensive fine-tuning.

**References:**

- Paper: "Med-Flamingo: A Multimodal Medical Few-shot Learner (Various, 2023)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a medical image using Med-Flamingo's gated cross-attention pipeline. |

