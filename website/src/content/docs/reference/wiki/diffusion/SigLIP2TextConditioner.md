---
title: "SigLIP2TextConditioner<T>"
description: "SigLIP 2 text encoder conditioning module (Tschannen et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

SigLIP 2 text encoder conditioning module (Tschannen et al. 2025).
Improves on SigLIP via additional captioning loss + MAP head; the
text-encoder body remains a standard CLIP-style stack.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPretrained(SigLIP2Variant,String,String)` | Loads a paper-canonical SigLIP 2 conditioner with its real pretrained tokenizer from HuggingFace. |

