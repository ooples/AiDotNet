---
title: "ImageFolderDataset<T>"
description: "Loads images from a directory structure where each subdirectory is a class label."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Vision`

Loads images from a directory structure where each subdirectory is a class label.

## For Beginners

Organize your images into folders named after the classes,
and this loader handles the rest:

## How It Works

Mirrors PyTorch's `torchvision.datasets.ImageFolder`. Expects a directory structure:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageFolderDataset(ImageFolderDatasetOptions)` | Creates a new ImageFolderDataset with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Gets the class names discovered from directory names. |
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `NumClasses` | Gets the number of classes. |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

