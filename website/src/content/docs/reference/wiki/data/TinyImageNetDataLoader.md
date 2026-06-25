---
title: "TinyImageNetDataLoader<T>"
description: "Loads the Tiny ImageNet 200-class image-classification dataset (500 train + 50 val + 50 test per class at 64×64)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Tiny ImageNet 200-class image-classification dataset
(500 train + 50 val + 50 test per class at 64×64).

## How It Works

Expects the canonical Stanford CS231n directory layout:

Auto-download fetches the canonical zip from cs231n.stanford.edu.
Validation labels are mapped via `val_annotations.txt`; test
labels are zeroed since they're hidden upstream.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TinyImageNetDataLoader(TinyImageNetDataLoaderOptions)` | Creates a new Tiny ImageNet data loader. |

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

