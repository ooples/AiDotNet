---
title: "EntityLinking<T>"
description: "EntityLinking - Entity extraction and linking for document text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Postprocessing.Document`

EntityLinking - Entity extraction and linking for document text.

## For Beginners

Documents contain references to people, places,
organizations, and other entities. This tool identifies them:

- Extract named entities (people, places, organizations)
- Link entities to canonical forms
- Resolve entity references
- Build entity relationships

Key features:

- Named entity recognition
- Entity disambiguation
- Reference resolution
- Relationship extraction

Example usage:

## How It Works

EntityLinking identifies named entities in document text and links them
to canonical representations or external knowledge bases.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EntityLinking` | Creates a new EntityLinking instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverse` | Entity linking does not support inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAlias(String,String)` | Adds an alias for an existing entity. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the entity linker. |
| `ExtractDates(String)` | Extracts dates from text. |
| `ExtractEmails(String)` | Extracts email addresses from text. |
| `ExtractMoneyAmounts(String)` | Extracts money amounts from text. |
| `ExtractOrganizations(String)` | Extracts organization names from text. |
| `ExtractPersons(String)` | Extracts person names from text. |
| `ExtractPhoneNumbers(String)` | Extracts phone numbers from text. |
| `ExtractUrls(String)` | Extracts URLs from text. |
| `LinkEntity(Entity)` | Links an entity to its canonical form. |
| `ProcessCore(String)` | Extracts all entities from the text. |
| `RegisterEntity(Entity,IEnumerable<String>)` | Registers a known entity with optional aliases. |
| `ValidateInput(String)` | Validates the input text. |

