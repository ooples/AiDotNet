---
title: "LLM2CLIPOptions"
description: "Configuration options for the LLM2CLIP model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.Encoders`

Configuration options for the LLM2CLIP model.

## For Beginners

LLM2CLIP upgrades CLIP's text understanding by replacing its simple text model
with a powerful language model (like ChatGPT's text engine). This means it can understand complex
descriptions, nuanced language, and longer text much better than regular CLIP.

## How It Works

LLM2CLIP (Huang et al., 2024) from Microsoft enhances CLIP's text encoder by replacing it with
an LLM (such as LLaMA or Mistral). The LLM text embeddings provide richer semantic understanding,
especially for complex captions and long-form text. The LLM is fine-tuned with contrastive learning
to align with the existing CLIP vision encoder.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LLM2CLIPOptions` | Initializes default LLM2CLIP options. |
| `LLM2CLIPOptions(LLM2CLIPOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FreezeVisionEncoder` | Gets or sets whether to freeze the vision encoder during LLM alignment. |
| `LLMBackbone` | Gets or sets the LLM backbone name for the text encoder. |
| `LLMHiddenDim` | Gets or sets the LLM hidden dimension. |
| `LoRARank` | Gets or sets the LoRA rank for efficient fine-tuning. |
| `LossType` | Gets or sets the contrastive loss type. |
| `UseLoRA` | Gets or sets whether to use LoRA for efficient LLM fine-tuning. |

