---
title: "ImageClassificationDataset<T>"
description: "An in-memory image classification dataset with an optional composable transform pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision`

An in-memory image classification dataset with an optional composable transform pipeline.

## For Beginners

If you have images already loaded as arrays/tensors and want
to apply normalization or other transforms:

## How It Works

Holds image tensors and class labels in memory with optional transforms applied on access.
Use this when you already have image data as tensors and want to attach a transform pipeline.
For loading from disk, use `ImageFolderDataset` instead.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageClassificationDataset(Tensor<>[],Int32[],ITransform<[],[]>)` | Creates an in-memory image classification dataset from tensors. |
| `ImageClassificationDataset([][],Int32[],Int32,Int32,Int32,ITransform<[],[]>)` | Creates an in-memory image classification dataset. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets the number of channels. |
| `Description` |  |
| `FeatureCount` |  |
| `ImageHeight` | Gets the image height. |
| `ImageWidth` | Gets the image width. |
| `Name` |  |
| `NumClasses` | Gets the number of classes in the dataset. |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

