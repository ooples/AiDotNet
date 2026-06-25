---
title: "CoresetSelector"
description: "Selects a representative coreset from a dataset using distance-based strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Selects a representative coreset from a dataset using distance-based strategies.

## How It Works

Coreset selection finds a small representative subset that approximates the full dataset.
Supports greedy facility location, k-Center, and random selection strategies.
Works on pre-computed feature embeddings (distance matrix).

## Methods

| Method | Summary |
|:-----|:--------|
| `Select(Double[0:,0:])` | Selects coreset indices from pre-computed pairwise distances. |
| `Select(Double[][])` | Selects coreset indices from embedding vectors using Euclidean distance. |

