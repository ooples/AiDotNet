---
title: "SquadDataLoader<T>"
description: "Loads the SQuAD question answering dataset (100K+ Q&A pairs on Wikipedia articles)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the SQuAD question answering dataset (100K+ Q&A pairs on Wikipedia articles).

## How It Works

SQuAD expects JSON files:

Features are concatenated context + question tokens Tensor[N, MaxContextLength + MaxQuestionLength].
Labels are answer start/end positions Tensor[N, 2].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SquadDataLoader(SquadDataLoaderOptions)` | Creates a new SQuAD data loader. |

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
| `CharOffsetToTokenIndex(String,Int32)` | Converts a character offset in text to an approximate token index by counting word boundaries (whitespace-separated tokens). |
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

