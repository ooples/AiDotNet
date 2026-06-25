---
title: "IDEFICS<T>"
description: "IDEFICS: 80B open reproduction of Flamingo for interleaved image-text generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

IDEFICS: 80B open reproduction of Flamingo for interleaved image-text generation.

## For Beginners

IDEFICS is an 80 billion parameter open-source model that
replicates DeepMind's Flamingo architecture. It processes interleaved sequences of images
and text using gated cross-attention layers inserted into a LLaMA decoder, with a perceiver
resampler compressing visual features. Trained on the OBELICS web-scraped dataset, it
excels at few-shot multimodal tasks. Default values follow the original paper settings.

## How It Works

IDEFICS (Laurencon et al., 2023) is an 80B parameter open-source model that reproduces
the Flamingo architecture. It uses an OpenCLIP ViT-H vision encoder, a perceiver resampler
for visual token compression, and gated cross-attention layers interleaved within a
LLaMA-based decoder for multimodal text generation.

**References:**

- Paper: "OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents" (Laurencon et al., NeurIPS 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using IDEFICS's 80B Flamingo-style architecture. |
| `GetExtraTrainableLayers` |  |

