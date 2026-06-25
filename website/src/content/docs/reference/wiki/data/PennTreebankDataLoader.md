---
title: "PennTreebankDataLoader<T>"
description: "Loads the Penn Treebank language modeling dataset (Mikolov-preprocessed split)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the Penn Treebank language modeling dataset (Mikolov-preprocessed split).

## How It Works

Expects:

Auto-download fetches the canonical Mikolov tarball.
Features are input token sequences Tensor[N, SequenceLength].
Labels are next-token targets Tensor[N, SequenceLength] (shifted by 1).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PennTreebankDataLoader(PennTreebankDataLoaderOptions)` | Creates a new PTB data loader. |

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
| `LoadRawTextAsync(DatasetSplit,CancellationToken)` | Loads the raw, unprocessed text content for the requested PTB split, auto-downloading via `AutoDownload` if the file is not already cached. |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

