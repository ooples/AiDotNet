---
title: "InputOutputDataLoaderBase<T, TInput, TOutput>"
description: "Abstract base class for input-output data loaders providing common supervised learning functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Loaders`

Abstract base class for input-output data loaders providing common supervised learning functionality.

## For Beginners

This base class handles common input-output data operations:

- Storing features (X) and labels (Y) for supervised learning
- Splitting data into training, validation, and test sets
- Shuffling data to improve training
- Iterating through data in batches

Concrete implementations (CsvDataLoader, ImageDataLoader) extend this
to load specific data formats.

## How It Works

InputOutputDataLoaderBase provides shared implementation for all supervised learning data loaders including:

- Feature (X) and label (Y) data management
- Train/validation/test splitting
- Shuffling and batching capabilities
- Progress tracking through ICountable

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InputOutputDataLoaderBase(Int32)` | Initializes a new instance of the InputOutputDataLoaderBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` |  |
| `FeatureCount` |  |
| `Features` |  |
| `HasNext` |  |
| `IsShuffled` |  |
| `Labels` |  |
| `OutputDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSplitSizes(Int32,Double,Double)` | Computes split sizes from ratios and total count. |
| `ExtractBatch(Int32[])` | Extracts a batch of features and labels at the specified indices. |
| `GetBatches(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>)` |  |
| `GetBatchesAsync(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>,Int32,CancellationToken)` |  |
| `GetNextBatch` |  |
| `InitializeIndices(Int32)` | Initializes indices array after data is loaded. |
| `OnReset` |  |
| `Shuffle(Nullable<Int32>)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `TryGetNextBatch(ValueTuple<,>)` |  |
| `Unshuffle` |  |
| `ValidateSplitRatios(Double,Double)` | Validates split ratios. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Indices` | Indices for current data ordering (used for shuffling). |
| `LoadedFeatures` | Storage for loaded feature data. |
| `LoadedLabels` | Storage for loaded label data. |
| `NumOps` | Numeric operations helper for type T. |

