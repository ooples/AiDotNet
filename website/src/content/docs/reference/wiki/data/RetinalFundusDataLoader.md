---
title: "RetinalFundusDataLoader<T>"
description: "Loads retinal fundus photography datasets for diabetic retinopathy detection (5-class grading)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads retinal fundus photography datasets for diabetic retinopathy detection (5-class grading).

## How It Works

Retinal Fundus expects:

The CSV has columns: image (filename without extension), level (0-4 severity grade).
Labels are one-hot encoded as Tensor[N, 5] for the 5 DR severity levels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RetinalFundusDataLoader(RetinalFundusDataLoaderOptions)` | Creates a new Retinal Fundus data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `OutputDimension` |  |
| `SeverityNames` | Gets the severity level names. |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

