---
title: "BigEarthNetDataLoader<T>"
description: "Loads the BigEarthNet multi-label remote sensing dataset (590K Sentinel-2 patches, 19 or 43 classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the BigEarthNet multi-label remote sensing dataset (590K Sentinel-2 patches, 19 or 43 classes).

## How It Works

BigEarthNet expects:

Each patch directory contains GeoTIFF band files and a JSON metadata file with CORINE Land Cover labels.
Labels are multi-hot encoded as Tensor[N, NumClasses].
For RGB mode (NumBands=3), only B04(R), B03(G), B02(B) are loaded.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BigEarthNetDataLoader(BigEarthNetDataLoaderOptions)` | Creates a new BigEarthNet data loader. |

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
| `BuildClassNameIndex` | Builds a mapping from BigEarthNet CORINE Land Cover class names to indices. |
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

