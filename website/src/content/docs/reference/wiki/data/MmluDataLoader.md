---
title: "MmluDataLoader<T>"
description: "Loads MMLU — Massive Multitask Language Understanding (Hendrycks et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads MMLU — Massive Multitask Language Understanding (Hendrycks et al. 2021).

## How It Works

Expects the canonical AI release layout:

Each CSV has columns `question, A, B, C, D, answer` with no header.
Auto-downloads the canonical Hendrycks tarball. 4-way multiple choice
shape: features [N, 4, MaxQuestionLength], one-hot labels [N, 4].

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

