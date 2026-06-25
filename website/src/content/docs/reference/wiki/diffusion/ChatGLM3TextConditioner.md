---
title: "ChatGLM3TextConditioner<T>"
description: "ChatGLM3 text encoder conditioning module (Zeng et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

ChatGLM3 text encoder conditioning module (Zeng et al. 2023).
Pre-LN RMSNorm Transformer stack with RoPE multi-query attention
(KV heads = 1) and SiLU FFN. Used in Kolors and other Chinese-language
diffusion pipelines.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPretrained(ChatGLM3Variant,String,String)` | Loads a paper-canonical ChatGLM3 conditioner with its real pretrained SentencePiece tokenizer from HuggingFace. |

