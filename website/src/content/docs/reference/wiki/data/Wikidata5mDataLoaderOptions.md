---
title: "Wikidata5mDataLoaderOptions"
description: "Configuration options for the Wikidata5M knowledge graph data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Graph`

Configuration options for the Wikidata5M knowledge graph data loader.

## How It Works

Wikidata5M is a large-scale knowledge graph with ~5M entities and ~21M triplets.
Supports link prediction and entity classification tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `EmbeddingDimension` | Entity embedding dimension. |
| `MaxSamples` | Optional maximum number of triplets to load. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

