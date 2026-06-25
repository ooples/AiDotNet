---
title: "Emu3<T>"
description: "Emu3: next-token prediction unifies understanding and generation in a single model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

Emu3: next-token prediction unifies understanding and generation in a single model.

## For Beginners

Emu3 simplifies multimodal AI by treating everything — text and
images — as sequences of tokens predicted one at a time. Images are converted to discrete
tokens via a VQVAE codebook, then mixed with text tokens in a unified vocabulary. A single
autoregressive transformer handles both understanding and generation without needing separate
diffusion or contrastive modules. Default values follow the original paper settings.

## How It Works

Emu3 (Wang et al., 2024) simplifies the multimodal architecture by using next-token prediction
as the sole training objective for both understanding and generation. Images are tokenized into
discrete visual tokens via a VQVAE, then interleaved with text tokens in a unified vocabulary.
A single autoregressive transformer generates both text and visual tokens.

**References:**

- Paper: "Emu3: Next-Token Prediction is All You Need" (Wang et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates using Emu3's next-token prediction architecture. |
| `GetExtraTrainableLayers` |  |

