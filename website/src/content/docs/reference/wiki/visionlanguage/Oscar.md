---
title: "Oscar<T>"
description: "Oscar (Object-Semantics Aligned pre-training) using object tags as anchor points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Foundational`

Oscar (Object-Semantics Aligned pre-training) using object tags as anchor points.

## For Beginners

Oscar is a vision-language model. Default values follow the original paper settings.

## How It Works

Oscar (Li et al., ECCV 2020) uses detected object tags as "anchor points" to ease cross-modal
alignment. The input to the transformer is a triple of (word tokens, object tags, region features),
processed in a single BERT stream. Pre-training uses masked token loss and contrastive loss.

**References:**

- Paper: "Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks" (Li et al., ECCV 2020)

