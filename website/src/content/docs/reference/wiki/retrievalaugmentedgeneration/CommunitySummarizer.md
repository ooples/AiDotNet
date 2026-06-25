---
title: "CommunitySummarizer<T>"
description: "Generates structured summaries for detected communities in a knowledge graph."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Communities`

Generates structured summaries for detected communities in a knowledge graph.

## For Beginners

After finding communities, this class describes what each one is about.
For example, a community containing "Einstein", "Bohr", "Planck" connected by "collaborated_with"
and "influenced" relations might be summarized as:
"Physics pioneers community: 3 entities centered around Einstein, with key relations: collaborated_with, influenced"

## How It Works

For each community, the summarizer collects member entities, identifies central entities
by degree centrality, finds dominant relation types, and generates a structured description.

## Methods

| Method | Summary |
|:-----|:--------|
| `Summarize(KnowledgeGraph<>,LeidenResult,Int32,Int32)` | Generates summaries for all communities in a Leiden result. |
| `SummarizePartition(KnowledgeGraph<>,Dictionary<String,Int32>,Int32,Int32,Int32)` | Generates summaries for a specific partition (community assignment) at a given hierarchy level. |

