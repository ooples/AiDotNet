---
title: "OptimizationDataBatcher<T, TInput, TOutput>"
description: "Provides batch iteration utilities for optimization input data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Optimizers`

Provides batch iteration utilities for optimization input data.

## For Beginners

When training machine learning models, you typically don't
feed all your data at once. Instead, you break it into smaller "batches" and train
on each batch. This class makes that process easy and efficient:

## How It Works

OptimizationDataBatcher provides efficient batch iteration over optimization input data,
integrating with the DataLoader batching infrastructure for consistent behavior.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OptimizationDataBatcher(OptimizationInputData<,,>,Int32,Boolean,Boolean,Nullable<Int32>,IDataSampler)` | Initializes a new instance of the OptimizationDataBatcher class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets the batch size. |
| `DataSize` | Gets the total number of samples in the training data. |
| `NumBatches` | Gets the number of batches per epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CastToDataType()` | Casts a source type to a target type using implicit boxing. |
| `ExtractBatch(Int32[])` | Extracts a batch of data given the indices. |
| `GetBatchIndices` | Iterates through training data in batches, returning only the indices. |
| `GetBatches` | Iterates through training data in batches. |
| `GetIndices` | Gets all indices for iteration, optionally shuffled. |
| `SelectRows(,Int32[])` | Selects rows from an input structure at the specified indices. |
| `WithClassBalancing(IReadOnlyList<Int32>,Int32)` | Creates a new batcher with weighted sampling for class balancing. |
| `WithCurriculumLearning(IEnumerable<>,Int32,CurriculumStrategy)` | Creates a new batcher with curriculum learning. |
| `WithSampler(IDataSampler)` | Creates a new batcher with a different sampler. |

