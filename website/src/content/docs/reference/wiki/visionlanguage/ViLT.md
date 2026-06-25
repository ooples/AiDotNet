---
title: "ViLT<T>"
description: "ViLT (Vision-and-Language Transformer) with minimal architecture using patch embeddings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

ViLT (Vision-and-Language Transformer) with minimal architecture using patch embeddings.

## For Beginners

ViLT is a vision-language model. Default values follow the original paper settings.

## How It Works

ViLT (Kim et al., ICML 2021) removes the CNN/object detector entirely, linearly embedding raw
image patches and concatenating them with text tokens in a single transformer. This makes it
60x faster than region-feature-based models at comparable accuracy. Pre-trained with ITM and MLM.

**References:**

- Paper: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision" (Kim et al., ICML 2021)

