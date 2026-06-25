---
title: "Gsm8kDataLoader<T>"
description: "Loads the GSM8K grade-school math word-problem dataset (Cobbe et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the GSM8K grade-school math word-problem dataset (Cobbe et al. 2021).

## How It Works

Expects:

Each line is `{"question": "...", "answer": "... #### N"}`.
Auto-download fetches the canonical OpenAI release. Features are
tokenized question sequences Tensor[N, MaxQuestionLength]; labels are
tokenized answer (chain-of-thought + final number) sequences
Tensor[N, MaxAnswerLength].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Gsm8kDataLoader(Gsm8kDataLoaderOptions)` | Creates a new GSM8K data loader. |

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

