---
title: "Cifar100DataLoader<T>"
description: "Loads the CIFAR-100 image classification dataset (50k train / 10k test, 32x32 RGB, 100 fine / 20 coarse classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the CIFAR-100 image classification dataset (50k train / 10k test, 32x32 RGB, 100 fine / 20 coarse classes).

## How It Works

CIFAR-100 has 100 fine-grained classes grouped into 20 superclasses. Each sample is stored as
2 bytes (coarse label, fine label) + 3072 bytes of pixel data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Cifar100DataLoader(Cifar100DataLoaderOptions)` | Creates a new CIFAR-100 data loader. |

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

