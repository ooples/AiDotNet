---
title: "LeafFederatedDataLoader<T>"
description: "Data loader that reads LEAF benchmark JSON splits and exposes both aggregated (X, Y) data and per-client partitions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Data loader that reads LEAF benchmark JSON splits and exposes both aggregated (X, Y) data and per-client partitions.

## For Beginners

Use this loader when you want to run federated learning with realistic per-user splits
provided by LEAF datasets.

## How It Works

LEAF is a federated learning benchmark suite where each user corresponds to one client.
This loader preserves that structure through `ClientData` while also providing
aggregated `Features`/`Labels`
for compatibility with the standard training facade.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeafFederatedDataLoader(String,String,LeafFederatedDatasetLoadOptions)` | Initializes a new instance of the `LeafFederatedDataLoader` class from LEAF JSON files. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientData` |  |
| `ClientIdToUserId` | Gets the mapping from internal client IDs (0..N-1) to original LEAF user IDs. |
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `OutputDimension` |  |
| `TestSplit` | Gets the loaded optional test split (one dataset per LEAF user). |
| `TotalCount` |  |
| `TrainSplit` | Gets the loaded training split (one dataset per LEAF user). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

