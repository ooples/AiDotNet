---
title: "ArcDataLoader<T>"
description: "Loads the AI2 Reasoning Challenge (ARC) multiple-choice science QA benchmark (Clark et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the AI2 Reasoning Challenge (ARC) multiple-choice science QA
benchmark (Clark et al. 2018) — Easy or Challenge variant.

## How It Works

Expects:

Auto-download fetches the canonical AllenAI release. Handles both
4-way and 5-way questions (ARC has a small fraction of 5-choice items);
the 5th choice is treated as an out-of-distribution distractor when
truncated. Features Tensor[N, 4, MaxSequenceLength]; one-hot labels
Tensor[N, 4].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ArcDataLoader(ArcDataLoaderOptions)` | Creates a new ARC data loader. |

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

