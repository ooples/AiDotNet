---
title: "DataVersionControlBase<T>"
description: "Base class for data version control implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.DataVersionControl`

Base class for data version control implementations.

## How It Works

**For Beginners:** This abstract base class provides common functionality for data
version control systems. It handles storage path management, hash computation for
integrity verification, and data lineage tracking.

Key features:

- Path security validation
- SHA-256 hash computation for data integrity
- Dataset versioning support
- Lineage tracking for reproducibility

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataVersionControlBase(String,String)` | Initializes a new instance of the DataVersionControlBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareDatasetVersions(String,String,String)` | Compares two dataset versions to see what changed. |
| `ComputeDatasetHash(String)` | Computes and stores a hash of the dataset for integrity verification. |
| `CreateDatasetSnapshot(String,Dictionary<String,String>,String)` | Creates a snapshot of multiple related datasets together. |
| `CreateDatasetVersion(String,String,String,Dictionary<String,Object>,Dictionary<String,String>)` | Creates a new dataset version. |
| `DeleteDatasetVersion(String,String)` | Deletes a specific dataset version. |
| `DeserializeFromJson(String)` | Deserializes a JSON string to an object. |
| `EnsureStorageDirectoryExists` | Ensures the storage directory exists. |
| `GetAllDatasetsInSnapshot(String)` | Retrieves all datasets in a multi-dataset snapshot. |
| `GetDatasetByTag(String,String)` | Gets a dataset version by its tag. |
| `GetDatasetDirectoryPath(String)` | Gets the directory path for a dataset. |
| `GetDatasetForRun(String)` | Gets the dataset version used by a specific training run. |
| `GetDatasetLineage(String,String)` | Gets the lineage information for a dataset version. |
| `GetDatasetSnapshot(String)` | Retrieves a dataset snapshot, returning only the first dataset in the snapshot. |
| `GetDatasetStatistics(String,String)` | Gets statistics about a dataset version. |
| `GetDatasetVersion(String,String)` | Retrieves a specific version of a dataset. |
| `GetLatestDatasetVersion(String)` | Gets the latest version of a dataset. |
| `GetRunsUsingDataset(String,String)` | Gets all training runs that used a specific dataset version. |
| `GetSanitizedFileName(String)` | Sanitizes a file name to prevent path traversal attacks. |
| `GetSanitizedPath(String,String)` | Gets a sanitized path, ensuring it doesn't escape the base directory. |
| `LinkDatasetToRun(String,String,String,String)` | Links a dataset version to a model training run. |
| `ListDatasetVersions(String)` | Lists all versions of a dataset. |
| `ListDatasets(String,Dictionary<String,String>)` | Lists all tracked datasets. |
| `RecordDatasetLineage(String,String,DatasetLineage)` | Records metadata about how a dataset was created or transformed. |
| `SerializeToJson(Object)` | Serializes an object to JSON. |
| `TagDatasetVersion(String,String,String)` | Tags a dataset version for easy reference. |
| `ValidateDatasetName(String)` | Validates that a dataset name is valid. |
| `ValidatePathWithinDirectory(String,String)` | Validates that a path is within the specified directory. |
| `VerifyDatasetIntegrity(String,String,String)` | Verifies that a dataset hasn't been modified by comparing its hash. |

## Fields

| Field | Summary |
|:-----|:--------|
| `JsonSettings` | JSON serialization settings for consistent serialization. |
| `StorageDirectory` | The directory where version control data is stored. |
| `SyncLock` | Lock object for thread-safe operations. |

