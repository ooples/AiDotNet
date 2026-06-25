---
title: "GraphRAGMode"
description: "Specifies the retrieval mode for enhanced GraphRAG."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the retrieval mode for enhanced GraphRAG.

## For Beginners

These modes represent different strategies for querying a knowledge graph:

- Local: Start from specific entities and explore their neighborhoods (best for specific questions)
- Global: Use community summaries for broad, thematic answers (best for "what are all X?" questions)
- Drift: Start broad with community summaries, then iteratively refine locally (best for complex queries)

## Fields

| Field | Summary |
|:-----|:--------|
| `Drift` | DRIFT search: starts with global community summaries, then iteratively refines via local exploration. |
| `Global` | Global search: community-level retrieval using pre-computed community summaries. |
| `Local` | Local search: entity-centric retrieval via graph traversal from matched entities. |

