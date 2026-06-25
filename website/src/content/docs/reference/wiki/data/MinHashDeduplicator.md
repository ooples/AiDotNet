---
title: "MinHashDeduplicator"
description: "Detects near-duplicate documents using MinHash with Locality-Sensitive Hashing (LSH)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Detects near-duplicate documents using MinHash with Locality-Sensitive Hashing (LSH).

## How It Works

MinHash approximates Jaccard similarity between document shingle sets.
LSH banding reduces comparison count from O(n^2) to near-linear.
Commonly used for deduplicating web crawl data (e.g., C4, The Pile).

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSignature(String)` | Computes the MinHash signature for a document. |
| `DeterministicHash(String,Int32,Int32)` | FNV-1a hash for deterministic, cross-process-stable hashing of character shingles. |
| `EstimateSimilarity(Int32[],Int32[])` | Estimates Jaccard similarity between two documents from their MinHash signatures. |
| `FindDuplicates(IReadOnlyList<String>)` | Finds duplicate indices from a collection of documents using LSH banding. |

