---
title: "ParquetDataLoader<T>"
description: "Reads tabular data from Apache Parquet columnar files using the Parquet.Net library."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

Reads tabular data from Apache Parquet columnar files using the Parquet.Net library.

## For Beginners

Parquet files are commonly produced by Apache Spark, Pandas, and other
data processing frameworks. This loader reads them directly into AiDotNet for training.

## How It Works

Parquet is an efficient columnar storage format widely used in data engineering and ML pipelines.
This loader reads Parquet files and converts numeric columns into tensors for training.
Supports all standard Parquet compression codecs (Snappy, GZIP, ZSTD, LZ4, Brotli) and encodings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParquetDataLoader(ParquetDataLoaderOptions)` | Creates a new Parquet data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `OutputDimension` |  |
| `ResolvedFeatureColumns` | Gets the column names used as features. |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

