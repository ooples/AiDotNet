---
title: "SigLIPTextConditioner<T>"
description: "SigLIP text encoder conditioning module (Zhai et al., ICCV 2023)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

SigLIP text encoder conditioning module (Zhai et al., ICCV 2023).
Same encoder architecture as CLIP; the paper's contribution is the
sigmoid contrastive loss (vs CLIP's softmax), which is upstream of
this encoder body.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromPretrained(SigLIPVariant,String,String)` | Loads a paper-canonical SigLIP text conditioner with its real pretrained tokenizer from HuggingFace. |

