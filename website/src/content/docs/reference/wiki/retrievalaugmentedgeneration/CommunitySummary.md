---
title: "CommunitySummary"
description: "Represents a summary of a detected community within a knowledge graph."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Communities`

Represents a summary of a detected community within a knowledge graph.

## For Beginners

After detecting communities, each one gets a summary describing:

- Which entities belong to it
- What the most important entities are (by connection count)
- What types of relationships dominate
- A human-readable description of what the community represents

## Properties

| Property | Summary |
|:-----|:--------|
| `CommunityId` | Unique identifier for this community. |
| `Description` | Structured text description of the community's content and themes. |
| `EntityIds` | IDs of all entities belonging to this community. |
| `KeyEntities` | IDs of the most central/important entities in the community (by degree centrality). |
| `KeyRelations` | Most frequent relation types within the community. |
| `Level` | Hierarchy level at which this community was detected (0 = finest). |

