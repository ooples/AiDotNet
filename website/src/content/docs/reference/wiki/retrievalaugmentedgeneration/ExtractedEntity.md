---
title: "ExtractedEntity"
description: "Represents an entity extracted from text during knowledge graph construction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Construction`

Represents an entity extracted from text during knowledge graph construction.

## For Beginners

When building a knowledge graph from text, the first step is
identifying entities — the "things" mentioned in the text (people, places, organizations).
Each extracted entity has:

- Name: The text mention ("Albert Einstein")
- Label: The entity type ("PERSON", "ORGANIZATION", "LOCATION")
- Confidence: How sure we are this is a real entity (0.0 to 1.0)
- Offsets: Where in the text this entity was found

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Confidence score for this extraction (0.0 to 1.0). |
| `EndOffset` | End character offset in the source text. |
| `Label` | Entity type label (e.g., PERSON, ORGANIZATION, LOCATION, CONCEPT). |
| `Name` | The entity's surface form (name as it appears in text). |
| `StartOffset` | Start character offset in the source text. |

