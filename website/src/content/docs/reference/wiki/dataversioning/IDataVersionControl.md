---
title: "IDataVersionControl"
description: "Interface for data version control systems."
section: "API Reference"
---

`Interfaces` · `AiDotNet.DataVersioning`

Interface for data version control systems.

## How It Works

**For Beginners:** Data version control (like DVC) helps you track
changes to your datasets, ensuring you can always reproduce experiments
with the exact same data.

Key concepts:

- Dataset: A collection of data files (training data, validation data, etc.)
- Version: A specific snapshot of a dataset at a point in time
- Hash: A unique identifier based on file contents
- Lineage: The history of how data was transformed

## Properties

| Property | Summary |
|:-----|:--------|
| `StorageDirectory` | Gets the storage directory for data versions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddVersion(String,String,String,Dictionary<String,String>)` | Adds a new version of a dataset. |
| `CompareVersions(String,String,String)` | Compares two versions of a dataset. |
| `CreateDataset(String,String,Dictionary<String,String>)` | Registers a new dataset for version control. |
| `DeleteDataset(String)` | Deletes a dataset and all its versions. |
| `DeleteVersion(String,String)` | Deletes a specific version. |
| `GetDataPath(String,String)` | Gets the path to access a specific version's data. |
| `GetLineage(String,String)` | Gets the lineage (data ancestry) for a version. |
| `GetVersion(String,String)` | Gets a specific version of a dataset. |
| `ListDatasets` | Lists all registered datasets. |
| `ListVersions(String)` | Lists all versions of a dataset. |
| `RecordLineage(String,String,List<ValueTuple<String,String>>,String,Dictionary<String,Object>)` | Records data lineage (transformation history). |

