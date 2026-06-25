---
title: "Cifar10DataLoader<T>"
description: "Loads the CIFAR-10 image classification dataset (50k train / 10k test, 32x32 RGB, 10 classes)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the CIFAR-10 image classification dataset (50k train / 10k test, 32x32 RGB, 10 classes).

## How It Works

CIFAR-10 contains 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck. The dataset is stored in a binary format where each sample
is 1 byte label + 3072 bytes of pixel data (32x32x3 in CHW order: 1024 red, 1024 green, 1024 blue).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Cifar10DataLoader(Cifar10DataLoaderOptions)` | Creates a new CIFAR-10 data loader. |

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

