---
title: "LXMERT<T>"
description: "LXMERT (Learning Cross-Modality Encoder Representations from Transformers) with three-encoder architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

LXMERT (Learning Cross-Modality Encoder Representations from Transformers) with three-encoder architecture.

## For Beginners

LXMERT is a vision-language model. Default values follow the original paper settings.

## How It Works

LXMERT (Tan and Bansal, EMNLP 2019) uses three transformer encoders: an object relationship
encoder for visual features, a language encoder for text, and a cross-modality encoder that
performs cross-attention between the two modalities.

**References:**

- Paper: "LXMERT: Learning Cross-Modality Encoder Representations from Transformers" (Tan and Bansal, EMNLP 2019)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetExtraTrainableLayers` |  |

