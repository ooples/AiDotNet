---
title: "DualTextConditioner<T>"
description: "Dual text encoder conditioning module combining CLIP and T5 encoders."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

Dual text encoder conditioning module combining CLIP and T5 encoders.

## For Beginners

This uses TWO text encoders together for better results.

Why two encoders?

- CLIP: Great at understanding the visual "gist" of your prompt

(e.g., knowing what a cat looks like, what "golden light" means visually)

- T5: Great at understanding language details

(e.g., counting objects, understanding spatial relationships, rendering text)

How they work together:

1. CLIP produces a "pooled" embedding (one vector summarizing the whole prompt)

→ Used for the global style/content of the image

2. T5 produces "sequence" embeddings (one vector per token)

→ Used for cross-attention, giving fine-grained control

Used by:

- FLUX.1: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
- SD3: CLIP ViT-L/14 (768-dim) + OpenCLIP ViT-bigG/14 (1280-dim) + T5-XXL (4096-dim)
- Imagen: T5-XXL (text only, no CLIP)

## How It Works

Many modern diffusion models use multiple text encoders together to get the best
of each encoder's strengths. This module combines a CLIP encoder (for pooled embeddings
and visual-semantic alignment) with a T5 encoder (for detailed text understanding
and cross-attention).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DualTextConditioner(CLIPTextConditioner<>,T5TextConditioner<>)` | Initializes a new dual text encoder conditioning module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CLIPEmbeddingDimension` | Gets the CLIP encoder's embedding dimension (for pooled embeddings). |
| `ConditioningType` |  |
| `EmbeddingDimension` | Gets the combined context dimension (T5 embedding dimension for cross-attention). |
| `MaxSequenceLength` | Gets the maximum sequence length (uses T5's longer sequence length). |
| `ProducesPooledOutput` | Gets whether this module produces pooled output (yes, from CLIP). |
| `T5EmbeddingDimension` | Gets the T5 encoder's embedding dimension (for cross-attention). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Tensor<>)` |  |
| `EncodeDual(String)` | Encodes text using both CLIP and T5 encoders. |
| `EncodeText(Tensor<>,Tensor<>)` |  |
| `FromPretrained(CLIPVariant,T5Variant,String)` | Loads a paper-canonical SD3-style dual encoder with its real pretrained tokenizers from HuggingFace (CLIP + T5). |
| `GetPooledEmbedding(Tensor<>)` |  |
| `GetUnconditionalDual(Int32)` | Gets unconditional embeddings from both encoders. |
| `GetUnconditionalEmbedding(Int32)` |  |
| `Tokenize(String)` |  |
| `TokenizeBatch(String[])` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_clipEncoder` | The CLIP text encoder providing pooled and sequence embeddings. |
| `_t5Encoder` | The T5 text encoder providing high-dimensional sequence embeddings. |

