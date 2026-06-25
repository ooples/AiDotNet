---
title: "ChestXray14DataLoader<T>"
description: "Loads the NIH Chest X-ray 14 multi-label classification dataset (112K images, 14 disease labels)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the NIH Chest X-ray 14 multi-label classification dataset (112K images, 14 disease labels).

## How It Works

Chest X-ray 14 expects:

The CSV contains columns: Image Index, Finding Labels, Follow-up #, Patient ID, Patient Age, etc.
Finding Labels is a pipe-separated list of disease names (e.g., "Atelectasis|Effusion").
Labels are multi-hot encoded as Tensor[N, 14] where each dimension corresponds to a disease.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChestXray14DataLoader(ChestXray14DataLoaderOptions)` | Creates a new ChestX-ray14 data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `DiseaseNames` | Gets the disease label names. |
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

