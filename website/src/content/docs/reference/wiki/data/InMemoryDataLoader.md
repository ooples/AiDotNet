---
title: "InMemoryDataLoader<T, TInput, TOutput>"
description: "A simple in-memory data loader for supervised learning data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

A simple in-memory data loader for supervised learning data.

## For Beginners

This is the easiest data loader to use. Simply pass your
feature data (X) and label data (Y) to the constructor, and you're ready to train!

**Example:**
```cs
// Create feature matrix and label vector
var features = new Matrix<double>(100, 5); // 100 samples, 5 features
var labels = new Vector<double>(100); // 100 labels

// Create the loader
var loader = new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(features, labels);

// Use with AiModelBuilder
var result = await builder
.ConfigureDataLoader(loader)
.ConfigureModel(model)
.BuildAsync();
```

## How It Works

InMemoryDataLoader is the simplest way to create a data loader from existing data.
It's ideal for:

- Small to medium datasets that fit in memory
- Quick prototyping and testing
- Converting raw arrays or matrices to the IDataLoader interface

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InMemoryDataLoader(,)` | Creates a new in-memory data loader with the specified features and labels. |

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
| `CastToDataType()` | Casts a source type to a target type using implicit boxing. |
| `CreateSubsetLoader(Int32[])` | Creates a new loader containing only the data at the specified indices. |
| `ExtractBatch(Int32[])` |  |
| `ExtractFeatures(Int32[])` | Extracts features at the specified indices. |
| `ExtractLabels(Int32[])` | Extracts labels at the specified indices. |
| `GetInputDimensions()` | Gets dimensions from the input features. |
| `GetOutputDimensions()` | Gets dimensions from the output labels. |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

