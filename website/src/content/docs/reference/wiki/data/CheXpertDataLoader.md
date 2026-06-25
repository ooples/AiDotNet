---
title: "CheXpertDataLoader<T>"
description: "Loads the CheXpert chest radiograph dataset (224K images, 14 observations with uncertainty)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the CheXpert chest radiograph dataset (224K images, 14 observations with uncertainty).

## How It Works

CheXpert expects:

The CSV has columns: Path, Sex, Age, Frontal/Lateral, AP/PA, then 14 observation columns.
Observation values: 1.0 (positive), 0.0 (negative), -1.0 (uncertain), blank (not mentioned).
Labels are stored as Tensor[N, 14] with uncertainty handled per the UncertaintyPolicy option.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CheXpertDataLoader(CheXpertDataLoaderOptions)` | Creates a new CheXpert data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `ObservationNames` | Gets the observation label names. |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

