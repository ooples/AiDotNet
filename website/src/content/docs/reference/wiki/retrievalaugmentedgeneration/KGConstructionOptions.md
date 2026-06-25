---
title: "KGConstructionOptions"
description: "Configuration options for automated knowledge graph construction from text."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Construction`

Configuration options for automated knowledge graph construction from text.

## For Beginners

These settings control how the KG construction pipeline works:

- MaxChunkSize: How many characters per text chunk (smaller = more precise entity extraction)
- ChunkOverlap: Overlap between chunks to avoid missing entities at boundaries
- EntityConfidenceThreshold: Minimum confidence to accept an extracted entity
- EnableEntityResolution: Whether to merge similar entity names (e.g., "Einstein" and "A. Einstein")
- EntitySimilarityThreshold: How similar two entity names must be to merge them

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkOverlap` | Character overlap between adjacent chunks. |
| `EnableEntityResolution` | Whether to merge similar entity names via string similarity. |
| `EntityConfidenceThreshold` | Minimum confidence threshold to accept an extracted entity. |
| `EntitySimilarityThreshold` | Minimum string similarity (0-1) for merging entity names. |
| `MaxChunkSize` | Maximum characters per text chunk for entity extraction. |
| `MaxEntitiesPerSentence` | Maximum number of entities per sentence for co-occurrence relation generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ValidateCrossFieldConstraints` | Validates cross-field constraints. |

