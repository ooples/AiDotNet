---
title: "GemmaTextConditioner<T>"
description: "Gemma text encoder conditioning module (Gemma Team 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

Gemma text encoder conditioning module (Gemma Team 2024).
Pre-LN RMSNorm Transformer stack with RoPE multi-head attention and
SiLU FFN. Used by Imagen 3 and other Google diffusion pipelines.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPretrained(GemmaVariant,String,String)` | Loads a paper-canonical Gemma conditioner with its real pretrained SentencePiece tokenizer from HuggingFace. |
| `GetPooledEmbedding(Tensor<>)` | Decoder-style models pool by extracting the embedding at the last non-pad token position. |

