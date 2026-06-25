---
title: "WikiText2DataLoader<T>"
description: "Loads the WikiText-2 language modeling dataset (≈ 2M tokens train, ≈ 245k val, ≈ 281k test from Wikipedia articles)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the WikiText-2 language modeling dataset (≈ 2M tokens train, ≈ 245k val, ≈ 281k test
from Wikipedia articles).

## How It Works

WikiText-2 expects:

Features are input token sequences Tensor[N, SequenceLength].
Labels are next-token target sequences Tensor[N, SequenceLength] (shifted by 1).

Uses the `wikitext-2-raw-v1` archive flavor (preserves original casing
and punctuation), which is the modern standard for subword-tokenizer
pipelines. The companion `WikiText103DataLoader` uses the
older tokenized flavor for backward compatibility — pick whichever matches
your tokenizer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WikiText2DataLoader(WikiText2DataLoaderOptions)` | Creates a new WikiText-2 data loader. |

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

