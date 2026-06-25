---
title: "PaLI<T>"
description: "PaLI (Pathways Language and Image model): large-scale ViT-e vision encoder + mT5 text decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

PaLI (Pathways Language and Image model): large-scale ViT-e vision encoder + mT5 text decoder.

## For Beginners

PaLI (Pathways Language and Image model) from Google combines
the largest dense ViT encoder (ViT-e with 4 billion parameters) with a multilingual mT5
text encoder-decoder. Visual tokens are linearly projected and prepended to text input,
then the mT5 model processes the combined sequence. Its multilingual training enables
vision-language tasks across 100+ languages. Default values follow the original paper
settings.

## How It Works

PaLI (Chen et al., ICLR 2023) combines a ViT-e vision encoder with an mT5 text encoder-decoder.
Visual tokens from the ViT are linearly projected and prepended to the text input tokens.
The mT5 encoder-decoder then processes the mixed visual-text sequence to generate text output.

**References:**

- Paper: "PaLI: A Jointly-Scaled Multilingual Language-Image Model" (Chen et al., ICLR 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using PaLI's visual-token prepending architecture. |

