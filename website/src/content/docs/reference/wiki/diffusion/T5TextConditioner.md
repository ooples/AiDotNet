---
title: "T5TextConditioner<T>"
description: "T5 text encoder conditioning module (Raffel et al., JMLR 2020)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

T5 text encoder conditioning module (Raffel et al., JMLR 2020).
Used as the conditioning encoder for Stable Diffusion 3, FLUX.1, and
Imagen pipelines. Pre-LN RMSNorm stack with learned relative position
bias (paper-shared across all encoder layers).

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPretrained(T5Variant,String,String)` | Loads a paper-canonical T5 conditioner with its real pretrained SentencePiece tokenizer from HuggingFace. |
| `GetPooledEmbedding(Tensor<>)` | T5 pools by mean over non-pad tokens. |

