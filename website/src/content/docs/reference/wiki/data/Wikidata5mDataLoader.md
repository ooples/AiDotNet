---
title: "Wikidata5mDataLoader<T>"
description: "Loads Wikidata5M knowledge graph triplets as tensor features and labels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Graph`

Loads Wikidata5M knowledge graph triplets as tensor features and labels.

## How It Works

Expects TSV files with triplets (head, relation, tail):

Features are entity pair embeddings Tensor[N, 2 * EmbeddingDimension].
Labels are relation index Tensor[N, 1].

