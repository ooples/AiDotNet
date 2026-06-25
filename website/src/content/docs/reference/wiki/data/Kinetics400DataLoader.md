---
title: "Kinetics400DataLoader<T>"
description: "Loads the Kinetics-400 human action recognition dataset (~300K clips, 400 classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Video.Benchmarks`

Loads the Kinetics-400 human action recognition dataset (~300K clips, 400 classes).

## How It Works

Kinetics-400 expects pre-extracted frames:

Features are frame tensors Tensor[N, FramesPerVideo * FrameHeight * FrameWidth * 3].
Labels are one-hot Tensor[N, 400].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Kinetics400DataLoader(Kinetics400DataLoaderOptions)` | Creates a new Kinetics-400 data loader. |

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

