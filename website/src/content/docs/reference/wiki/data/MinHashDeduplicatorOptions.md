---
title: "MinHashDeduplicatorOptions"
description: "Configuration options for MinHash-based near-duplicate detection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for MinHash-based near-duplicate detection.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumBands` | Number of bands for LSH banding. |
| `NumHashFunctions` | Number of hash functions for the MinHash signature. |
| `Seed` | Random seed for reproducibility. |
| `ShingleSize` | N-gram size for shingling. |
| `SimilarityThreshold` | Jaccard similarity threshold for duplicate detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

