---
title: "SuperGlueDataLoader<T>"
description: "Loads SuperGLUE benchmark sub-tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads SuperGLUE benchmark sub-tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC).

## How It Works

SuperGLUE expects JSONL files:

Features are token-index encoded text Tensor[N, MaxSequenceLength].
Labels are one-hot Tensor[N, NumClasses].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SuperGlueDataLoader(SuperGlueDataLoaderOptions)` | Creates a new SuperGLUE data loader. |

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

