---
title: "Qwen2TextConditioner<T>"
description: "Qwen2 text encoder conditioning module (Yang et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

Qwen2 text encoder conditioning module (Yang et al. 2024).
Pre-LN RMSNorm Transformer stack with RoPE grouped-query attention and
SiLU FFN. Used in multilingual diffusion pipelines for stronger Asian-
language prompt understanding.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPretrained(Qwen2Variant,String,String)` | Loads a paper-canonical Qwen2 conditioner with its real pretrained BPE tokenizer from HuggingFace. |

