---
title: "StanfordCarsDataLoader<T>"
description: "Loads the Stanford Cars dataset (Krause et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Stanford Cars dataset (Krause et al. 2013) — 196 fine-grained car classes.

## How It Works

Expects (manually-extracted):

MAT annotations parsed via MatFileHandler; `class` field is 1-indexed (1..196).

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

