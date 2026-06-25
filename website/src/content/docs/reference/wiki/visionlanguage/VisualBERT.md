---
title: "VisualBERT<T>"
description: "VisualBERT: single-stream transformer that concatenates visual and text tokens."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

VisualBERT: single-stream transformer that concatenates visual and text tokens.

## For Beginners

VisualBERT is a vision-language model. Default values follow the original paper settings.

## How It Works

VisualBERT (Li et al., 2019) concatenates Faster R-CNN region features with text token embeddings
and processes them in a single BERT-style transformer, enabling implicit cross-modal alignment
through self-attention over the combined sequence.

**References:**

- Paper: "VisualBERT: A Simple and Performant Baseline for Vision and Language" (Li et al., 2019)

