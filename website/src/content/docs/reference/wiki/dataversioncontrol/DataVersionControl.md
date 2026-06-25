---
title: "DataVersionControl<T>"
description: "Implementation of data version control for tracking dataset changes over time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DataVersionControl`

Implementation of data version control for tracking dataset changes over time.

## How It Works

**For Beginners:** This is a complete implementation of data version control that manages
the lifecycle of your datasets, similar to how Git manages code.

Features include:

- Dataset versioning with hash-based integrity verification
- Linking datasets to training runs for reproducibility
- Tagging versions for easy reference
- Lineage tracking for data provenance
- Multi-dataset snapshots for experiment reproducibility

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataVersionControl(String)` | Initializes a new instance of the DataVersionControl class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareDatasetVersions(String,String,String)` | Compares two dataset versions to see what changed. |
| `CreateDatasetSnapshot(String,Dictionary<String,String>,String)` | Creates a snapshot of multiple related datasets together. |
| `CreateDatasetVersion(String,String,String,Dictionary<String,Object>,Dictionary<String,String>)` | Creates a new dataset version. |
| `DeleteDatasetVersion(String,String)` | Deletes a specific dataset version. |
| `GetAllDatasetsInSnapshot(String)` | Retrieves all datasets in a multi-dataset snapshot. |
| `GetDatasetByTag(String,String)` | Gets a dataset version by its tag. |
| `GetDatasetForRun(String)` | Gets the dataset version used by a specific training run. |
| `GetDatasetLineage(String,String)` | Gets the lineage information for a dataset version. |
| `GetDatasetSnapshot(String)` | Retrieves a dataset snapshot, returning only the first dataset in the snapshot. |
| `GetDatasetStatistics(String,String)` | Gets statistics about a dataset version. |
| `GetDatasetVersion(String,String)` | Retrieves a specific version of a dataset. |
| `GetLatestDatasetVersion(String)` | Gets the latest version of a dataset. |
| `GetRunsUsingDataset(String,String)` | Gets all training runs that used a specific dataset version. |
| `LinkDatasetToRun(String,String,String,String)` | Links a dataset version to a model training run. |
| `ListDatasetVersions(String)` | Lists all versions of a dataset. |
| `ListDatasets(String,Dictionary<String,String>)` | Lists all tracked datasets. |
| `RecordDatasetLineage(String,String,DatasetLineage)` | Records metadata about how a dataset was created or transformed. |
| `TagDatasetVersion(String,String,String)` | Tags a dataset version for easy reference. |

