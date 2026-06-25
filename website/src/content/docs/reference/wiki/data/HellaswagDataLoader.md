---
title: "HellaswagDataLoader<T>"
description: "Loads the HellaSwag 4-way commonsense NLI benchmark (Zellers et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the HellaSwag 4-way commonsense NLI benchmark (Zellers et al. 2019).

## How It Works

Expects:

Auto-download fetches the canonical Rowan Zellers GitHub release.
Features Tensor[N, 4 * MaxSequenceLength] holds the 4 candidate
(context + ending) sequences concatenated; labels are one-hot
4-class vectors Tensor[N, 4]. Test split labels are filled with the
uniform prior since they aren't released publicly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HellaswagDataLoader(HellaswagDataLoaderOptions)` | Creates a new HellaSwag data loader. |

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
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

