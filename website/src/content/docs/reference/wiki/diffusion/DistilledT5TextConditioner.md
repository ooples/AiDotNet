---
title: "DistilledT5TextConditioner<T>"
description: "Distilled T5 text encoder conditioning module — same architecture as T5 but half the layer count, per the DistilBERT-style knowledge-distillation recipe (Sanh et al., 2019)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

Distilled T5 text encoder conditioning module — same architecture as T5
but half the layer count, per the DistilBERT-style knowledge-distillation
recipe (Sanh et al., 2019).

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPretrained(DistilledT5Variant,String,String)` | Loads a paper-canonical DistilledT5 conditioner with its real pretrained tokenizer (shares the T5 SentencePiece vocab from `google/t5-v1_1-base`). |

