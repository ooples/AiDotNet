---
title: "Imdb50kDataLoader<T>"
description: "Loads the IMDB 50k movie review sentiment analysis dataset (25k train / 25k test, binary classification)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the IMDB 50k movie review sentiment analysis dataset (25k train / 25k test, binary classification).

## How It Works

The IMDB dataset contains 50,000 movie reviews for binary sentiment classification (positive/negative).
Reviews are tokenized into word indices using a simple whitespace/punctuation tokenizer,
with a configurable vocabulary size and sequence length.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Imdb50kDataLoader(Imdb50kDataLoaderOptions)` | Creates a new IMDB 50k data loader. |

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

