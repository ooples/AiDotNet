---
title: "AgNewsDataLoader<T>"
description: "Loads the AG News topic classification dataset (4 classes: World, Sports, Business, Sci/Tech)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the AG News topic classification dataset (4 classes: World, Sports, Business, Sci/Tech).

## How It Works

Expects CSV files `train.csv` and `test.csv` with columns
`class_index,title,description`. Auto-download fetches the canonical
release from the FastAI mirror.
Features are tokenized title+description sequences Tensor[N, MaxSequenceLength].
Labels are one-hot 4-class vectors Tensor[N, 4].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgNewsDataLoader(AgNewsDataLoaderOptions)` | Creates a new AG News data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `LoadRawTextAsync(DatasetSplit,CancellationToken)` | Loads the raw, unprocessed CSV text content for the requested AG News split, auto-downloading via `AutoDownload` if the file is not already cached. |
| `ParseAgNewsCsv(String,List<String>,List<Int32>)` | Parses the AG News CSV format. |
| `ReadCsvField(String,Int32,StringBuilder)` | Reads one quoted CSV field handling RFC4180 double-quote escapes. |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

