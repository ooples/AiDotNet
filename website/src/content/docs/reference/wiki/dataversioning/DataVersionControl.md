---
title: "DataVersionControl"
description: "DVC-equivalent data version control system for ML datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DataVersioning`

DVC-equivalent data version control system for ML datasets.

## How It Works

**For Beginners:** This class provides Git-like version control for your datasets.
It tracks changes, maintains history, and ensures reproducibility.

Key features:

- Content-addressable storage (files identified by hash)
- Efficient deduplication (same content stored once)
- Full version history with diff capabilities
- Data lineage tracking
- Metadata and tagging support

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataVersionControl(String)` | Initializes a new instance of the DataVersionControl class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `StorageDirectory` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddVersion(String,String,String,Dictionary<String,String>)` |  |
| `CompareVersions(String,String,String)` |  |
| `CreateDataset(String,String,Dictionary<String,String>)` |  |
| `DeleteDataset(String)` |  |
| `DeleteVersion(String,String)` |  |
| `Dispose` | Disposes the data version control system. |
| `GetDataPath(String,String)` |  |
| `GetLineage(String,String)` |  |
| `GetLineageInternal(String,String,HashSet<String>)` | Internal helper for GetLineage that tracks visited nodes to prevent infinite recursion. |
| `GetVersion(String,String)` |  |
| `GetVersionInternal(List<DataVersion>,String,String)` | Internal version lookup helper that operates within an existing lock. |
| `ListDatasets` |  |
| `ListVersions(String)` |  |
| `RecordLineage(String,String,List<ValueTuple<String,String>>,String,Dictionary<String,Object>)` |  |

