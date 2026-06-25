---
title: "CLIPTextConditioner<T>"
description: "CLIP text encoder conditioning module (Radford et al., ICML 2021)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

CLIP text encoder conditioning module (Radford et al., ICML 2021).
Primary text conditioner for Stable Diffusion 1.x / 2.x and one of two
encoders in SDXL / SD3 / FLUX.1.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLIPTextConditioner(ITokenizer,CLIPVariant,NeuralNetworkArchitecture<>)` | Constructs a CLIP text conditioner with an explicit paper-canonical tokenizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | CLIP parameter count = layer-stack params + the post-pool projection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildDefaultArchitecture(CLIPVariant)` | PyTorch-style lazy architecture: token-ID inputs are rank-2 `[batch, seqLen]`. |
| `FromPretrained(CLIPVariant,String,String)` | Loads a paper-canonical CLIP text conditioner with its real pretrained HuggingFace tokenizer. |
| `GetParameters` |  |
| `GetPooledEmbedding(Tensor<>)` | CLIP pools by extracting the embedding at the EOS token position (Radford 2021 §3.1) and then applying `_textProjection` to map hidden-dim → embedding-dim. |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_textProjection` | CLIP text_projection: a separate learnable hidden→embedding linear projection applied ONLY to the EOS-pooled output (Radford 2021 §3.1), NOT to every sequence position. |

