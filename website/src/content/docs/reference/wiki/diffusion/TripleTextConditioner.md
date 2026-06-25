---
title: "TripleTextConditioner<T>"
description: "Triple text encoder conditioning module combining two CLIP encoders and a T5 encoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Conditioning`

Triple text encoder conditioning module combining two CLIP encoders and a T5 encoder.

## For Beginners

This uses THREE text encoders together for the best possible results.

Why three encoders?

- CLIP ViT-L/14 (768-dim): Fast, general visual understanding of your prompt
- OpenCLIP ViT-bigG/14 (1280-dim): Larger, more detailed visual understanding
- T5-XXL (4096-dim): Deep language understanding for complex prompts

How they work together:

1. Both CLIP encoders produce "pooled" embeddings (one vector each summarizing the prompt)

→ These are concatenated into a combined 2048-dim (768+1280) vector
→ Used for the global conditioning vector fed to the MMDiT timestep embedder

2. T5 produces "sequence" embeddings (one vector per token, 4096-dim each)

→ Used for cross-attention, giving fine-grained control over details

Used by:

- Stable Diffusion 3 (SD3): All three encoders
- SD3 Turbo: All three encoders with fewer steps

## How It Works

Stable Diffusion 3 uses three text encoders to achieve the highest quality text understanding:
two CLIP encoders for visual-semantic alignment (with pooled embeddings) and a T5 encoder
for detailed language understanding (with cross-attention sequence embeddings).

**Reference:** Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TripleTextConditioner(CLIPTextConditioner<>,CLIPTextConditioner<>,T5TextConditioner<>)` | Initializes a new triple text encoder conditioning module for SD3. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CLIPGEmbeddingDimension` | Gets the second CLIP/OpenCLIP encoder's embedding dimension (1280 for ViT-bigG/14). |
| `CLIPLEmbeddingDimension` | Gets the first CLIP encoder's embedding dimension (768 for ViT-L/14). |
| `CombinedPooledDimension` | Gets the combined pooled embedding dimension (CLIP-L + CLIP-G = 768 + 1280 = 2048). |
| `ConditioningType` |  |
| `EmbeddingDimension` | Gets the T5 context dimension for cross-attention (4096 for T5-XXL). |
| `MaxSequenceLength` | Gets the maximum sequence length (uses T5's longer sequence length). |
| `ProducesPooledOutput` | Gets whether this module produces pooled output (yes, from both CLIP encoders). |
| `T5EmbeddingDimension` | Gets the T5 encoder's embedding dimension (4096 for T5-XXL). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConcatenatePooledEmbeddings(Tensor<>,Tensor<>)` | Concatenates pooled embeddings from CLIP-L and CLIP-G along the feature dimension. |
| `Encode(Tensor<>)` |  |
| `EncodeText(Tensor<>,Tensor<>)` |  |
| `EncodeTriple(String)` | Encodes text using all three encoders (CLIP-L, CLIP-G/OpenCLIP, and T5). |
| `FromPretrained(CLIPVariant,CLIPVariant,T5Variant,String)` | Loads a paper-canonical SDXL-style triple encoder with its real pretrained tokenizers from HuggingFace (CLIP-L + CLIP-G + T5). |
| `GetCombinedPooledEmbedding(String)` | Gets the combined pooled embedding from both CLIP encoders for the given text. |
| `GetPooledEmbedding(Tensor<>)` |  |
| `GetUnconditionalEmbedding(Int32)` |  |
| `GetUnconditionalTriple(Int32)` | Gets unconditional embeddings from all three encoders. |
| `Tokenize(String)` |  |
| `TokenizeBatch(String[])` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_clipGEncoder` | The second CLIP/OpenCLIP text encoder (ViT-bigG/14, 1280-dim). |
| `_clipLEncoder` | The first CLIP text encoder (ViT-L/14, 768-dim). |
| `_t5Encoder` | The T5 text encoder providing high-dimensional sequence embeddings. |

