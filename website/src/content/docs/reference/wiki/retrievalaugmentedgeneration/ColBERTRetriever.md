---
title: "ColBERTRetriever<T>"
description: "Retrieves documents using ColBERT's token-level late interaction mechanism (Khattab & Zaharia 2020, \"ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT\")."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Retrievers`

Retrieves documents using ColBERT's token-level late interaction mechanism
(Khattab & Zaharia 2020, "ColBERT: Efficient and Effective Passage Search
via Contextualized Late Interaction over BERT").

## For Beginners

Think of ColBERT like a detailed word-by-word
comparison. For each word in your query, ColBERT finds the single
best-matching word in each candidate document, then sums those best-match
scores. The result is a document score that captures whether
*every* aspect of your query has SOME good match in the document,
not just the overall topic.

## How It Works

ColBERT represents queries and documents as *multiple* contextualised
token embeddings rather than a single vector. The scoring rule
(paper §3.2 Eq. 1) is
`Score(Q, D) = Σ_q max_d cos(E_q, E_d)`
— for every query token, take the maximum cosine similarity against any
document token, then sum across query tokens. This "MaxSim" formulation
gives finer-grained matching than single-vector dense retrieval at a
fraction of the cost of full cross-encoder rerankers.

Production usage requires an `IColBertEmbedder` that
embeds query and document strings into per-token tensor banks. Without an
embedder, the retriever throws `NotSupportedException`:
there is no defensible "fallback" for ColBERT because the entire point
of the architecture is the contextual token-level representation. A
lexical-overlap stand-in would silently produce a different relevance
signal under the same class name, which is exactly the kind of silent
behavioural divergence this codebase rejects.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColBERTRetriever(IDocumentStore<>,String,Int32,Int32,IColBertEmbedder<>)` | Initializes a new instance of the ColBERTRetriever class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RetrieveCore(String,Int32,Dictionary<String,Object>)` | Retrieves documents using ColBERT MaxSim scoring (paper §3.2). |

