---
title: "MnistDataLoader<T>"
description: "Loads the MNIST handwritten digit classification dataset (60k train / 10k test, 28x28 grayscale)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision.Benchmarks`

Loads the MNIST handwritten digit classification dataset (60k train / 10k test, 28x28 grayscale).

## For Beginners

This is the easiest way to get started with image classification:

## How It Works

MNIST is the "Hello World" of machine learning - a dataset of handwritten digits (0-9).
Each image is 28x28 pixels, grayscale, with a corresponding digit label.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MnistDataLoader(MnistDataLoaderOptions)` | Creates a new MNIST data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `NumClasses` | Number of classes (digits 0-9). |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

