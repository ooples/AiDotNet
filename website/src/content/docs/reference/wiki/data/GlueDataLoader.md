---
title: "GlueDataLoader<T>"
description: "Loads GLUE benchmark sub-tasks (CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads GLUE benchmark sub-tasks (CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI).

## How It Works

GLUE expects TSV files:

Features are token-index encoded text Tensor[N, MaxSequenceLength].
Labels are one-hot Tensor[N, NumClasses] (binary for most tasks, 3-class for MNLI, regression for STS-B).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlueDataLoader(GlueDataLoaderOptions)` | Creates a new GLUE data loader. |

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

