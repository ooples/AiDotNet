---
title: "TruthfulQaDataLoader<T>"
description: "Loads the TruthfulQA benchmark (Lin et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the TruthfulQA benchmark (Lin et al. 2022) — 817 truthfulness questions
across 38 categories. Generation-style: question → best_answer.

## How It Works

Expects `{DataPath}/TruthfulQA.csv`. Auto-download fetches the
canonical sylinrl/TruthfulQA GitHub release. CSV columns:
`Type,Category,Question,Best Answer,...`.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ParseCsvRfc4180(String)` | RFC4180-complete CSV parser: walks the entire input character stream so quoted fields may contain commas, CR, LF, and escaped quotes. |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

