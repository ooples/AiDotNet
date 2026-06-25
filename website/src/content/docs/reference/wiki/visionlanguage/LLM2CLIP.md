---
title: "LLM2CLIP<T>"
description: "LLM2CLIP model enhancing CLIP's text encoder with LLM embeddings for richer semantic understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

LLM2CLIP model enhancing CLIP's text encoder with LLM embeddings for richer semantic understanding.

## For Beginners

LLM2CLIP supercharges CLIP by replacing its simple text encoder
with a powerful large language model (like LLaMA or Mistral). This gives the model much
richer text understanding — it can handle complex descriptions, long captions, and nuanced
queries that the original CLIP text encoder would struggle with. Default values follow
the original paper settings.

## How It Works

LLM2CLIP (Huang et al., 2024) replaces CLIP's text encoder with a fine-tuned LLM (e.g., LLaMA, Mistral),
providing much stronger text understanding for complex captions and long-form descriptions.

**References:**

- Paper: "LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation" (Huang et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

