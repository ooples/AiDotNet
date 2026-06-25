---
title: "KGConstructor<T>"
description: "Constructs a knowledge graph from unstructured text using heuristic entity and relation extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Construction`

Constructs a knowledge graph from unstructured text using heuristic entity and relation extraction.

## For Beginners

This class reads text and automatically builds a knowledge graph.

Given text: "Albert Einstein was born in Ulm, Germany. He worked at Princeton University."
It extracts:

- Entities: Albert Einstein (PERSON), Ulm (LOCATION), Germany (LOCATION), Princeton University (ORGANIZATION)
- Relations: Einstein BORN_IN Ulm, Einstein LOCATED_IN Germany, Einstein WORKED_AT Princeton University

The heuristic approach uses patterns like:

- Capitalized words = likely entity names
- Words like "born in", "works at", "located in" = relation indicators
- Entities appearing near each other = likely related

## How It Works

The construction pipeline operates in four stages:

1. **Chunk Text:** Split text into overlapping chunks for processing
2. **Extract Entities:** Identify named entities using regex patterns and capitalization heuristics
3. **Extract Relations:** Detect relations via co-occurrence and proximity-based patterns
4. **Entity Resolution:** Merge similar entity names to reduce duplicates

This implementation works without an external LLM (purely heuristic), providing a baseline
that can be extended or replaced with LLM-based extraction.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConstructFromText(String,KnowledgeGraph<>,KGConstructionOptions)` | Constructs a knowledge graph from the given text. |
| `ExtractEntities(String,Double)` | Extracts entities from a text chunk using heuristic patterns. |
| `ExtractRelations(String,List<ExtractedEntity>,Int32)` | Extracts relations between entities based on proximity and pattern matching. |

