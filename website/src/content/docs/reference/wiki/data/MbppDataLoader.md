---
title: "MbppDataLoader<T>"
description: "Loads the MBPP (Mostly Basic Python Problems) benchmark — 1,000 entry-level Python coding problems with natural-language descriptions and unit tests (Austin et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the MBPP (Mostly Basic Python Problems) benchmark — 1,000 entry-level
Python coding problems with natural-language descriptions and unit tests
(Austin et al. 2021).

## How It Works

Expects `{DataPath}/mbpp.jsonl`. Auto-download fetches the canonical
Google research GitHub release. Each record has `text` (problem),
`code` (canonical solution), `test_list`, and `task_id`.
Standard splits use task_id ranges: 1–10 prompts (skipped here), 11–510
test, 511–600 val, 601–974 train.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MbppDataLoader(MbppDataLoaderOptions)` | Creates a new MBPP data loader. |

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

