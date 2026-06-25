---
title: "IDEFICS3<T>"
description: "IDEFICS3: state-of-the-art 8B VLM trained with Docmatix for document understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Generative`

IDEFICS3: state-of-the-art 8B VLM trained with Docmatix for document understanding.

## For Beginners

IDEFICS3 builds on IDEFICS2 with improved training data,
including the Docmatix dataset for document understanding, and upgrades the language backbone
to Llama 3.1. It achieves state-of-the-art results on document QA benchmarks while being
trained exclusively on open datasets, making it one of the strongest open-source VLMs at
the 8B parameter scale. Default values follow the original paper settings.

## How It Works

IDEFICS3 (Laurencon et al., 2024) is an 8B parameter model that builds upon IDEFICS2 with
improved training data including the Docmatix dataset for document understanding. It retains
the SigLIP + perceiver + Llama 3.1 architecture but achieves state-of-the-art results on
document QA benchmarks while being trained exclusively on open datasets.

**References:**

- Paper: "Building and better understanding vision-language models: insights and future directions" (Laurencon et al., 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using IDEFICS3's SigLIP+Llama-3.1 architecture trained on Docmatix. |
| `GetExtraTrainableLayers` |  |

