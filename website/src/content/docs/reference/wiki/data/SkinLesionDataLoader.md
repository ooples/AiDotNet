---
title: "SkinLesionDataLoader<T>"
description: "Loads the ISIC Skin Lesion classification dataset (~25K images, 8 diagnostic categories)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the ISIC Skin Lesion classification dataset (~25K images, 8 diagnostic categories).

## How It Works

Skin Lesion expects:

The ground truth CSV has columns: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC.
Each column is 0.0 or 1.0, with exactly one positive per row (one-hot).
Labels are stored as Tensor[N, 8].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SkinLesionDataLoader(SkinLesionDataLoaderOptions)` | Creates a new Skin Lesion data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the class label names. |
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

