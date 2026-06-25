---
title: "WikiText103DataLoader<T>"
description: "Loads the WikiText-103 language modeling dataset (100M+ tokens from Wikipedia articles)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the WikiText-103 language modeling dataset (100M+ tokens from Wikipedia articles).

## How It Works

WikiText-103 expects:

Features are input token sequences Tensor[N, SequenceLength].
Labels are next-token target sequences Tensor[N, SequenceLength] (shifted by 1).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WikiText103DataLoader(WikiText103DataLoaderOptions)` | Creates a new WikiText-103 data loader. |

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
| `LoadRawTextAsync(DatasetSplit,CancellationToken)` | Loads the raw, unprocessed text content for the requested split, auto-downloading via `AutoDownload` if the file is not already cached. |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

