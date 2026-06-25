---
title: "Phi4Multimodal<T>"
description: "Phi-4-Multimodal: unified vision + audio + text in a single framework."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

Phi-4-Multimodal: unified vision + audio + text in a single framework.

## For Beginners

Phi-4-Multimodal from Microsoft extends the Phi-4 language model
to natively handle images, audio, and text in a single unified framework. Rather than having
separate models for different input types, Phi-4-Multimodal processes all modalities through
one architecture. It uses a SigLIP vision encoder with MLP projection for images and adds
audio understanding capabilities. This unified approach means a single model can answer
questions about photos, transcribe and understand speech, and process text — all without
switching between different specialized models. Default values follow the original paper
settings.

## How It Works

Phi-4-Multimodal (Microsoft, 2025) extends the Phi-4 language model with native support for
vision and audio inputs in a single unified framework. It uses a SigLIP vision encoder with
MLP projection to map visual features into the Phi-4 decoder space.

**References:**

- Paper: "Phi-4-Multimodal Technical Report" (Microsoft, 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using Phi-4-Multimodal's unified multi-modal architecture. |

