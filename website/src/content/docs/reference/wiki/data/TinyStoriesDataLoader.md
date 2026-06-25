---
title: "TinyStoriesDataLoader<T>"
description: "Loads the TinyStories synthetic LM corpus (Eldan & Li, 2023): ≈ 2.1M GPT-generated short stories with deliberately small vocabulary for small-scale language-model research."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the TinyStories synthetic LM corpus (Eldan & Li, 2023):
≈ 2.1M GPT-generated short stories with deliberately small vocabulary
for small-scale language-model research.

## How It Works

Expects:

Auto-download fetches both files from the canonical Hugging Face
repository `roneneldan/TinyStories`. Stories are concatenated and
chunked into fixed-length input/target sequences for LM training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TinyStoriesDataLoader(TinyStoriesDataLoaderOptions)` | Creates a new TinyStories data loader. |

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

