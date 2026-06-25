---
title: "Enwik8DataLoader<T>"
description: "Loads the enwik8 character-level Wikipedia language modeling benchmark (first 100M bytes of an English Wikipedia XML dump)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the enwik8 character-level Wikipedia language modeling benchmark
(first 100M bytes of an English Wikipedia XML dump).

## How It Works

Expects a single file `{DataPath}/enwik8`. Auto-download fetches the
canonical zip from mattmahoney.net. Operates byte-by-byte (no tokenization).
Standard split: first 90M chars → train, next 5M → val, last 5M → test.
Features are byte-id sequences Tensor[N, SequenceLength]; labels are
next-byte targets shifted by 1.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Enwik8DataLoader(Enwik8DataLoaderOptions)` | Creates a new enwik8 data loader. |

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
| `LoadRawTextAsync(DatasetSplit,CancellationToken)` | Loads the raw, unprocessed UTF-8 text content for the requested enwik8 split, auto-downloading via `AutoDownload` if the file is not already cached. |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

