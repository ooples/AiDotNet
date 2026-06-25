---
title: "ExactHashDeduplicator"
description: "Detects exact duplicate documents using cryptographic hashing (SHA-256)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Detects exact duplicate documents using cryptographic hashing (SHA-256).

## How It Works

The simplest and fastest deduplication method. Uses SHA-256 hashes to find
byte-identical documents. Optionally normalizes whitespace and case first.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeHash(String)` | Computes a hash for the given text. |
| `FindDuplicates(IReadOnlyList<String>)` | Finds duplicate indices from a collection of documents. |

