---
title: "IQueryEmbedder<T>"
description: "Single-vector query embedder contract for dense retrievers that need a per-query embedding (DPR-style, Karpukhin et al."
section: "API Reference"
---

`Interfaces` · `AiDotNet.RetrievalAugmentedGeneration.Retrievers`

Single-vector query embedder contract for dense retrievers that need
a per-query embedding (DPR-style, Karpukhin et al. 2020 §3.1). Concrete
implementations wrap a pretrained encoder (BERT-base / RoBERTa / E5 /
BGE / …) and produce a `Vector` in the same embedding
space as the document store the retriever is reading from.

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedQuery(String)` | Embeds `query` into a dense vector matching the retriever's document-store dimensionality. |

