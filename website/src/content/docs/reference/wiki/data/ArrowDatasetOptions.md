---
title: "ArrowDatasetOptions"
description: "Configuration options for Apache Arrow-based dataset access."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Formats`

Configuration options for Apache Arrow-based dataset access.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Number of rows per batch when reading. |
| `DataPath` | Path to the Arrow IPC file or directory of files. |
| `FeatureColumn` | Name of the feature column. |
| `LabelColumn` | Name of the label column. |
| `MemoryMap` | Whether to memory-map the file for large datasets. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

