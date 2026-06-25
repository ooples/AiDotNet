---
title: "PerplexityFilter"
description: "Filters documents based on perplexity scores from a simple n-gram language model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Filters documents based on perplexity scores from a simple n-gram language model.

## How It Works

Perplexity measures how "surprised" a language model is by a document.
Very high perplexity indicates gibberish or foreign language text.
Very low perplexity may indicate boilerplate or repetitive content.
Commonly used in C4 and CCNet data cleaning pipelines.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePerplexity(String)` | Computes the perplexity of a document under the trained language model. |
| `Filter(IReadOnlyList<String>)` | Filters documents by perplexity, returning indices of documents that should be removed. |
| `Train(IReadOnlyList<String>)` | Trains the n-gram language model on reference text. |

