---
title: "IColBertEmbedder<T>"
description: "Token-level embedder contract for ColBERT-style late interaction."
section: "API Reference"
---

`Interfaces` · `AiDotNet.RetrievalAugmentedGeneration.Retrievers`

Token-level embedder contract for ColBERT-style late interaction.
Concrete implementations wrap a pretrained ColBERT / ColBERTv2 / PLAID
model (typically loaded from ONNX) and expose two routines: one that
embeds the query into `[queryTokens, embedDim]` and one that
embeds a document into `[docTokens, embedDim]`. ColBERTRetriever
then computes the MaxSim score per query token against the document
token bank, summing across query tokens for the final relevance score
(Khattab & Zaharia 2020 §3.2).

## How It Works

Splitting query and document embedding into separate methods follows
the original ColBERT paper §3.2, which inserts distinct
`[Q]` / `[D]` marker tokens into the input so the
pretrained encoder can specialise its representations for the two
roles. Implementations should embed L2-normalised per-token vectors so
downstream cosine similarity reduces to a plain dot product.

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedDocument(String)` | Embeds a document into `[docTokens, embedDim]` L2-normalised vectors. |
| `EmbedQuery(String)` | Embeds a query into `[queryTokens, embedDim]` L2-normalised vectors. |

