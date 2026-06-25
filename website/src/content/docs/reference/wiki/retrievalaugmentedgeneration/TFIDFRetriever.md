---
title: "TFIDFRetriever<T>"
description: "TF-IDF (Term Frequency-Inverse Document Frequency) retrieval strategy with cached statistics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Retrievers`

TF-IDF (Term Frequency-Inverse Document Frequency) retrieval strategy with cached statistics.

## For Beginners

TF-IDF ranks documents by how unique and frequent terms are.

This implementation uses a cache to avoid recalculating term statistics on every search,
dramatically improving performance for repeated queries. The cache is automatically refreshed
when documents are added or removed.

## How It Works

Implements production-ready TF-IDF retrieval with intelligent caching to avoid recomputing
statistics on every query. The cache is automatically invalidated when the document count changes,
ensuring accuracy while maximizing performance.

