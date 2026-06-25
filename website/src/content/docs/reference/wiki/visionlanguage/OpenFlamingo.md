---
title: "OpenFlamingo<T>"
description: "OpenFlamingo: open-source reproduction of Flamingo with perceiver resampler."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

OpenFlamingo: open-source reproduction of Flamingo with perceiver resampler.

## For Beginners

OpenFlamingo is an open-source reproduction of DeepMind's
proprietary Flamingo model. It uses a perceiver resampler to compress visual features into
a fixed number of latent tokens, then injects them into an LLM decoder (LLaMA or MPT) via
gated cross-attention layers. This architecture excels at few-shot visual tasks — given a
few image-text examples, it can generalize to new queries. Default values follow the
original paper settings.

## How It Works

OpenFlamingo (Awadalla et al., 2023) replicates DeepMind's Flamingo architecture in an open-source
setting. It uses a CLIP ViT vision encoder, a perceiver resampler to compress visual features into
a fixed number of latent tokens, and gated cross-attention layers interleaved within an LLM decoder
(e.g., LLaMA, MPT) to condition text generation on visual information.

**References:**

- Paper: "OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models" (Awadalla et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using OpenFlamingo's open-source Flamingo architecture. |
| `GetExtraTrainableLayers` |  |

