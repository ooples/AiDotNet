---
title: "HumanEvalDataLoader<T>"
description: "Loads the HumanEval Python code-generation benchmark (Chen et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the HumanEval Python code-generation benchmark (Chen et al. 2021).

## How It Works

Expects `{DataPath}/HumanEval.jsonl`. Auto-download fetches the
canonical OpenAI release. Each record has `prompt`,
`canonical_solution`, `test`, `entry_point`; this loader
uses prompt as features and canonical_solution as the target. Scoring
against the unit tests requires running the model output through a
Python sandbox and is out of scope for the loader.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HumanEvalDataLoader(HumanEvalDataLoaderOptions)` | Creates a new HumanEval data loader. |

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

