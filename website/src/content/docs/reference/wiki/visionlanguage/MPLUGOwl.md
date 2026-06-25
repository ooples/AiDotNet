---
title: "MPLUGOwl<T>"
description: "mPLUG-Owl: modular VLM with visual abstractor for vision-language alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

mPLUG-Owl: modular VLM with visual abstractor for vision-language alignment.

## For Beginners

mPLUG-Owl from Alibaba uses a modular design where different
components can be trained independently. Its key innovation is the "visual abstractor" — a
module that sits between the vision encoder and language model, using learnable queries with
cross-attention to compress and align visual features into a format the language model can
understand. Think of it as a translator that converts raw image features into something
the text model can work with effectively. The modular design means you can upgrade
individual components without retraining the whole model. Default values follow the
original paper settings.

## How It Works

mPLUG-Owl (Alibaba, 2023) uses a modular architecture with a visual abstractor module that
learns to compress and align visual features from the ViT encoder before feeding them into
the LLaMA language model. The abstractor uses learnable queries with cross-attention.

**References:**

- Paper: "mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality" (Ye et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using mPLUG-Owl's modular architecture. |
| `GetExtraTrainableLayers` |  |

