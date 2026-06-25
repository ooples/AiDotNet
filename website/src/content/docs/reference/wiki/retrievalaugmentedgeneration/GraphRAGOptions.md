---
title: "GraphRAGOptions"
description: "Configuration options for the enhanced GraphRAG retrieval system."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Configuration options for the enhanced GraphRAG retrieval system.

## For Beginners

These settings control how GraphRAG queries the knowledge graph:

- Mode: Local (entity-focused), Global (community summaries), or Drift (global→local refinement)
- MaxHops: How far to traverse from matched entities in local mode
- CommunityDetection: Settings for the Leiden algorithm used in Global/Drift modes

## Properties

| Property | Summary |
|:-----|:--------|
| `CommunityDetection` | Options for Leiden community detection (used in Global and Drift modes). |
| `DriftMaxIterations` | Maximum iterations for DRIFT mode's local refinement phase. |
| `MaxHops` | Maximum traversal hops from matched entities in Local mode. |
| `Mode` | Retrieval mode. |

