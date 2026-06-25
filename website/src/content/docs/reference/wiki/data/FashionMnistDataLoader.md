---
title: "FashionMnistDataLoader<T>"
description: "Loads the Fashion-MNIST clothing classification dataset (60k train / 10k test, 28x28 grayscale)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the Fashion-MNIST clothing classification dataset (60k train / 10k test, 28x28 grayscale).

## How It Works

Fashion-MNIST is a drop-in replacement for MNIST with clothing images (t-shirt, trouser, pullover, etc.).
Same format (28x28 grayscale, 10 classes) but more challenging.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FashionMnistDataLoader(FashionMnistDataLoaderOptions)` | Creates a new Fashion-MNIST data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the human-readable class names. |
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

