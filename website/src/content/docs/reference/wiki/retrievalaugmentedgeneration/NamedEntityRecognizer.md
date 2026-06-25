---
title: "NamedEntityRecognizer"
description: "Production-ready Named Entity Recognition model using pattern matching and heuristics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.NER`

Production-ready Named Entity Recognition model using pattern matching and heuristics.

## For Beginners

This model identifies entities in text (people, places, organizations, etc.).

Think of it like highlighting important information in a document:

- Input: "Albert Einstein worked at Princeton University in New Jersey."
- Output: [Albert Einstein]=PERSON, [Princeton University]=ORGANIZATION, [New Jersey]=LOCATION

How it works:

1. Analyzes capitalization patterns
2. Looks for entity indicators (Corp, University, City, etc.)
3. Uses word lists of common names/places
4. Detects multi-word entities (first + last names, etc.)
5. Applies context rules (person names before "works at", etc.)

Entity types supported:

- PERSON: Names of people
- ORGANIZATION: Companies, institutions
- LOCATION: Cities, countries, places
- DATE: Temporal expressions

Future: Will be upgraded to BiLSTM-CRF neural network model for higher accuracy.

## How It Works

This NER model uses intelligent pattern matching, capitalization analysis, and context clues
to identify named entities. While not ML-based (yet), it provides production-ready accuracy
for common entity types through carefully crafted rules.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NamedEntityRecognizer` | Initializes a new instance of the `NamedEntityRecognizer` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractEntities(String)` | Extracts named entities from text. |

