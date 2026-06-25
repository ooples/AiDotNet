---
title: "CsvDataLoader<T>"
description: "Loads supervised learning data from CSV files into Matrix/Vector format."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Loads supervised learning data from CSV files into Matrix/Vector format.

## For Beginners

This loader reads CSV (Comma-Separated Values) files and converts them
into the format needed for training models. You specify which column contains the labels
(the values to predict), and the rest become features (input data).

## How It Works

**Example:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CsvDataLoader(String,Boolean,Int32,Int32)` | Creates a new CSV data loader. |

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
| `CreateSubsetLoader(Int32[])` | Creates a new InMemoryDataLoader containing only the data at the specified indices. |
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ParseCsvLine(String)` | Parses a single CSV line, handling RFC 4180 quoted fields (e.g., fields containing commas or quotes). |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

