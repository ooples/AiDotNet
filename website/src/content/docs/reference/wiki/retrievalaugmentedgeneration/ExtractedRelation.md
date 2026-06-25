---
title: "ExtractedRelation"
description: "Represents a relation extracted from text between two entities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Construction`

Represents a relation extracted from text between two entities.

## For Beginners

After finding entities in text, the next step is finding relationships
between them. For example, in "Einstein worked at Princeton University":

- SourceEntity: "Einstein" (PERSON)
- TargetEntity: "Princeton University" (ORGANIZATION)
- RelationType: "WORKED_AT"
- Confidence: 0.85 (fairly confident based on the verb "worked at")

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Confidence score for this relation extraction (0.0 to 1.0). |
| `RelationType` | The relation type (e.g., WORKS_AT, BORN_IN, PART_OF). |
| `SourceEntity` | The source (subject) entity name. |
| `TargetEntity` | The target (object) entity name. |

