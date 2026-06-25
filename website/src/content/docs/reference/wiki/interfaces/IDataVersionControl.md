---
title: "IDataVersionControl<T>"
description: "Defines the contract for data version control systems that track dataset changes over time."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for data version control systems that track dataset changes over time.

## How It Works

A data version control system manages versions of datasets used for training and evaluating models,
ensuring reproducibility and traceability.

**For Beginners:** Think of data version control like Git, but for your datasets instead of code.
Just like Git tracks changes to your code, data version control tracks changes to your data:

- Records what data was used to train each model
- Lets you go back to previous versions of datasets
- Helps reproduce experiments with exact same data
- Tracks where data came from and how it was transformed

Common scenarios include:

- Dataset updates (new examples added, errors corrected)
- Data preprocessing changes (different normalization, feature engineering)
- Train/validation/test splits that need to be reproduced
- Tracking data lineage for compliance

Why data version control matters:

- Models trained on different data versions perform differently
- Reproducing results requires exact same data
- Debugging requires knowing what data was used
- Compliance and auditing need data traceability
- Collaboration requires shared understanding of data versions

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareDatasetVersions(String,String,String)` | Compares two dataset versions to see what changed. |
| `ComputeDatasetHash(String)` | Computes and stores a hash of the dataset for integrity verification. |
| `CreateDatasetSnapshot(String,Dictionary<String,String>,String)` | Creates a snapshot of multiple related datasets together. |
| `CreateDatasetVersion(String,String,String,Dictionary<String,Object>,Dictionary<String,String>)` | Creates a new dataset version. |
| `DeleteDatasetVersion(String,String)` | Deletes a specific dataset version. |
| `GetDatasetByTag(String,String)` | Gets a dataset version by its tag. |
| `GetDatasetForRun(String)` | Gets the dataset version used by a specific training run. |
| `GetDatasetLineage(String,String)` | Gets the lineage information for a dataset version. |
| `GetDatasetSnapshot(String)` | Retrieves a dataset snapshot. |
| `GetDatasetStatistics(String,String)` | Gets statistics about a dataset version. |
| `GetDatasetVersion(String,String)` | Retrieves a specific version of a dataset. |
| `GetLatestDatasetVersion(String)` | Gets the latest version of a dataset. |
| `GetRunsUsingDataset(String,String)` | Gets all training runs that used a specific dataset version. |
| `LinkDatasetToRun(String,String,String,String)` | Links a dataset version to a model training run. |
| `ListDatasetVersions(String)` | Lists all versions of a dataset. |
| `ListDatasets(String,Dictionary<String,String>)` | Lists all tracked datasets. |
| `RecordDatasetLineage(String,String,DatasetLineage)` | Records metadata about how a dataset was created or transformed. |
| `TagDatasetVersion(String,String,String)` | Tags a dataset version for easy reference. |
| `VerifyDatasetIntegrity(String,String,String)` | Verifies that a dataset hasn't been modified by comparing its hash. |

