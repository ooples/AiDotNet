---
title: "EuroSatDataLoader<T>"
description: "Loads the EuroSAT land use/land cover classification dataset (27K patches, 64x64 RGB, 10 classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the EuroSAT land use/land cover classification dataset (27K patches, 64x64 RGB, 10 classes).

## How It Works

EuroSAT expects a folder-per-class structure:

Each class folder contains ~2,000-3,000 images. Labels are one-hot encoded as Tensor[N, 10].

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EuroSatDataLoader(EuroSatDataLoaderOptions)` | Creates a new EuroSAT data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the class names. |
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

