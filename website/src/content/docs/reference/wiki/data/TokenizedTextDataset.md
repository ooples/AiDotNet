---
title: "TokenizedTextDataset<T>"
description: "In-memory dataset of pre-tokenized text sequences for language model training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text`

In-memory dataset of pre-tokenized text sequences for language model training.

## For Beginners

After tokenizing your text (converting words to numbers),
use this dataset to hold the token sequences for training:

## How It Works

Stores token ID sequences where each sample is a fixed-length array of integer token IDs.
This is the standard input format for transformer-based language models (GPT, BERT, etc.).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TokenizedTextDataset(Int32[][],Int32[],Int32,Int32)` | Creates a tokenized text dataset from token ID sequences and labels. |

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

