---
title: "LLaVAMed<T>"
description: "LLaVA-Med: biomedical VLM with GPT-4-level visual understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Medical`

LLaVA-Med: biomedical VLM with GPT-4-level visual understanding.

## For Beginners

LLaVA-Med is a vision-language model for biomedical image
understanding and visual question answering. Default values follow the original paper
settings.

## How It Works

LLaVA-Med (Microsoft, 2023) adapts the LLaVA architecture for biomedical visual question
answering through curriculum learning on PubMed Central figure-caption pairs. It achieves
GPT-4-level visual understanding on biomedical images by first aligning visual and text
features on biomedical figure-caption data, then fine-tuning on biomedical VQA instruction
datasets for clinical and research applications.

**References:**

- Paper: "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine (Microsoft, 2023)"

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text from a biomedical image using LLaVA-Med's curriculum learning pipeline. |

