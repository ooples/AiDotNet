---
title: "TransformedDataLoader<T>"
description: "Wraps a data loader and applies a composable transform pipeline to feature data during batch extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Wraps a data loader and applies a composable transform pipeline to feature data during batch extraction.

## For Beginners

Use this to add normalization, scaling, or other preprocessing
to any existing data loader without modifying the original loader:

## How It Works

This wrapper applies an `ITransform` to every sample's feature
data when batches are retrieved. The underlying loader's data remains unmodified.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformedDataLoader(InputOutputDataLoaderBase<,Tensor<>,Tensor<>>,ITransform<[],[]>)` | Creates a transformed data loader wrapping the given inner loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FeatureCount` |  |
| `Features` |  |
| `HasNext` |  |
| `IsShuffled` |  |
| `Labels` |  |
| `Name` |  |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyTransformToTensor(Tensor<>)` | Applies the transform to each sample row in the tensor. |
| `GetBatches(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>)` |  |
| `GetBatchesAsync(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>,Int32,CancellationToken)` |  |
| `GetNextBatch` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Shuffle(Nullable<Int32>)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `TryGetNextBatch(ValueTuple<Tensor<>,Tensor<>>)` |  |
| `UnloadDataCore` |  |
| `Unshuffle` |  |

